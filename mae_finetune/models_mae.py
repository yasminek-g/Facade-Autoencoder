# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import numpy as np

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """
    Masked Autoencoder with a Vision Transformer backbone.

    This MAE class allows:
    - Random masking of patches
    - Block masking: masking a contiguous vertical or square block of patches
    - Combined masking: a large block plus random patches outside that block.

    It includes:
    - A ViT encoder (with optional pretrained/frozen weights)
    - A decoder that reconstructs masked patches
    - Methods for different masking strategies

    Recommended Usage for Stability:
    - Start training with random masking only (warmup epochs).
    - Gradually increase block_ratio and decrease random_ratio after warmup.
    - This approach stabilizes the loss and avoids sudden spikes.

    Args:
        img_size (int): Input image size (e.g., 224).
        patch_size (int): Patch size for splitting the image into patches.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        depth (int): Number of encoder blocks.
        num_heads (int): Number of attention heads in the encoder.
        decoder_embed_dim (int): Embedding dimension for the decoder.
        decoder_depth (int): Number of decoder blocks.
        decoder_num_heads (int): Number of attention heads in the decoder.
        mlp_ratio (float): MLP ratio in Transformer blocks.
        norm_layer: Normalization layer type.
        norm_pix_loss (bool): If True, use normalized pixel loss.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # pos_embed: positional embedding for all patches + cls token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Pretrained/frozen encoder blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        # Additional trainable blocks appended after the frozen encoder
        self.new_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(2)
        ])

        self.norm = norm_layer(embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # decoder_pred projects back to pixel space
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def no_weight_decay(self):
        """No weight decay for positional embeddings and mask tokens."""
        return {'pos_embed', 'cls_token', 'mask_token'}


    def initialize_weights(self):
        """Initialize model weights and positional embeddings."""
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        """Xavier initialization for Linear and constant init for LayerNorm."""
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)


    def freeze_pretrained_weights(self):
        """
        Freeze the weights of the pretrained encoder blocks.
        This is useful if you loaded pretrained MAE weights and only want to train added layers.
        """        
        for param in self.blocks.parameters():
            param.requires_grad = False
        print("Pretrained encoder blocks have been frozen.")


    def patchify(self, imgs):
        """
        Convert images into patch sequences.

        imgs: (N, 3, H, W)
        returns x: (N, L, patch_size**2 * 3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x


    def unpatchify(self, x):
        """
        Reconstruct images from patch sequences.

        x: (N, L, patch_size**2 *3)
        returns imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    

    def random_masking(self, x, mask_ratio=0.75):
        """
        Perform random masking by shuffling patches and keeping only a certain percentage.

        Args:
            x: [N, L, D]
            mask_ratio: fraction of patches to mask.

        Returns:
            x_masked: masked input features
            mask: binary mask (1=masked, 0=unmasked)
            ids_restore: restore indices to original order
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Random permutation of indices
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Select the patches to keep
        ids_keep = ids_shuffle[:, :len_keep]

        # Masked input
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate mask: 1 means masked, 0 means unmasked
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the mask aligned with original ordering
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def block_masking(self, x, block_ratio=0.5, flipped=True, random_offset=False):
        """
        Masks a contiguous vertical block of patches (full columns wide, block_rows high).
        If random_offset=True, vertical start row is chosen randomly.

        Args:
            x: [N, L, D]
            block_ratio: fraction of patch rows to mask (vertical block)
            flipped: if True, row=0 is bottom visually; else top is row=0.
            random_offset: if True, place block at a random vertical offset.

        Returns:
            x_masked, mask, ids_restore
        """
        N, L, D = x.shape
        h = w = int(L**0.5)
        block_rows = int(h * block_ratio)

        mask = torch.zeros(N, L, device=x.device)
        if random_offset:
            row_offset = torch.randint(low=0, high=h - block_rows + 1, size=(1,)).item()
            masked_row_start = row_offset
            masked_row_end = row_offset + block_rows
        else:
            if flipped:
                masked_row_start = 0
                masked_row_end = block_rows
            else:
                masked_row_start = h - block_rows
                masked_row_end = h

        for row in range(masked_row_start, masked_row_end):
            start_idx = row * w
            end_idx = (row + 1) * w
            mask[:, start_idx:end_idx] = 1

        # Reorder so unmasked come first
        unmasked_indices = []
        masked_indices = []
        for row in range(h):
            for col in range(w):
                idx = row*w + col
                if mask[0, idx] == 0:
                    unmasked_indices.append(idx)
                else:
                    masked_indices.append(idx)

        ordering = torch.tensor(unmasked_indices + masked_indices, device=x.device)
        ids_shuffle = ordering.unsqueeze(0).repeat(N,1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = len(unmasked_indices)
        x_masked = torch.gather(x, dim=1, index=ids_shuffle[:, :len_keep].unsqueeze(-1).expand(-1, -1, D))
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore
    

    def block_masking_bbox_bottom(self, x, block_ratio=0.3, horizontal_random_offset=True, flipped=True):
        """
        Masks a single large square block at the bottom.

        block_ratio: fraction of entire image to mask as a square block.
        We compute block_size = sqrt(block_ratio * L) and form a contiguous square block.

        If flipped=True, row=0 is bottom, block starts at bottom row.
        If flipped=False, anchor at the actual bottom (row=h-block_size).

        horizontal_random_offset: if True, block shifts horizontally at random.

        Returns x_masked, mask, ids_restore
        """
        N, L, D = x.shape
        h = w = int(L**0.5)

        # Determine block size from block_ratio
        block_area = int(block_ratio * L)
        block_size = int(block_area**0.5)
        block_size = min(block_size, h, w)

        if flipped:
            row_start, row_end = 0, block_size
        else:
            row_start, row_end = h - block_size, h

        if horizontal_random_offset:
            col_start = torch.randint(low=0, high=w - block_size + 1, size=(1,)).item()
        else:
            col_start = 0
        col_end = col_start + block_size

        mask = torch.zeros(N, L, device=x.device)
        # Fill the block area
        for row in range(row_start, row_end):
            start_idx = row * w + col_start
            end_idx = row * w + col_end
            mask[:, start_idx:end_idx] = 1

        # Reorder patches (unmasked first)
        unmasked_indices = []
        masked_indices = []
        for row in range(h):
            for col in range(w):
                idx = row*w + col
                if mask[0, idx] == 0:
                    unmasked_indices.append(idx)
                else:
                    masked_indices.append(idx)

        ordering = torch.tensor(unmasked_indices + masked_indices, device=x.device)
        ids_shuffle = ordering.unsqueeze(0).repeat(N,1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = len(unmasked_indices)
        x_masked = torch.gather(x, dim=1, index=ids_shuffle[:, :len_keep].unsqueeze(-1).expand(-1, -1, D))
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore


    def combined_masking(self, x, block_ratio=0.3, random_ratio=0.45, flipped=True, horizontal_random_offset=True, random_offset=False):
        """
        Combined masking: One big block + random patches outside the block.

        Steps:
        1) Mask a large square block (block_ratio * L patches).
        2) From the remaining unmasked patches, randomly mask random_ratio * L patches.
        No overlap between block and random masked areas.

        Ensure block_ratio + random_ratio <= 1.0 for a stable setup.

        For stability and decreasing loss:
        - Start training with random only (block_ratio=0, random_ratio=0.75).
        - Gradually increase block_ratio and decrease random_ratio during training epochs.
        - This progressive approach avoids sudden spikes in loss.

        Args:
            x: [N, L, D]
            block_ratio: fraction of entire image for block.
            random_ratio: fraction of entire image for random.
            flipped (bool): If True, row=0 is bottom visually.
            horizontal_random_offset (bool): If True, random horizontal offset for block.
            random_offset (bool): currently not used in this version for vertical offset.

        Returns:
            final_x_masked, combined_mask, final_ids_restore
        """
        N, L, D = x.shape
        total_patches = L

        # Block masking first
        x_block_masked, mask_block, ids_restore_block = self.block_masking_bbox_bottom(
            x, block_ratio=block_ratio, horizontal_random_offset=horizontal_random_offset, flipped=flipped
        )

        N, L_block, D = x_block_masked.shape
        if L_block == 0:
            # Entire image masked by block
            return x_block_masked, mask_block, ids_restore_block

        if block_ratio + random_ratio > 1.0:
            raise ValueError("block_ratio + random_ratio should not exceed 1.0 total masking.")

        num_to_mask_random = int(random_ratio * total_patches)
        if num_to_mask_random > L_block:
            raise ValueError(f"Not enough unmasked patches ({L_block}) to mask {num_to_mask_random} randomly.")

        # Randomly mask num_to_mask_random from L_block unmasked patches
        noise = torch.rand(N, L_block, device=x_block_masked.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep_local = ids_shuffle[:, :L_block - num_to_mask_random]
        ids_mask_local = ids_shuffle[:, L_block - num_to_mask_random:]

        x_final_masked = torch.gather(
            x_block_masked, dim=1,
            index=ids_keep_local.unsqueeze(-1).expand(-1, -1, D)
        )

        combined_mask = mask_block.clone()
        block_unmasked_positions = (combined_mask == 0)

        for i in range(N):
            global_unmasked_indices = torch.where(block_unmasked_positions[i])[0]
            to_mask_global = global_unmasked_indices[ids_mask_local[i]]
            combined_mask[i, to_mask_global] = 1

        final_x = []
        final_ids_restore_list = []
        for i in range(N):
            unmasked_i = torch.where(combined_mask[i] == 0)[0]
            masked_i   = torch.where(combined_mask[i] == 1)[0]
            ordering_i = torch.cat([unmasked_i, masked_i], dim=0)
            x_masked_final_i = torch.index_select(x[i], dim=0, index=unmasked_i)
            final_x.append(x_masked_final_i)
            ids_restore_i = torch.argsort(ordering_i)
            final_ids_restore_list.append(ids_restore_i.unsqueeze(0))

        final_x_masked = torch.nn.utils.rnn.pad_sequence(final_x, batch_first=True, padding_value=0.0)
        final_ids_restore = torch.cat(final_ids_restore_list, dim=0)

        return final_x_masked, combined_mask, final_ids_restore


    # def combined_masking(self, x, block_ratio=0.3, random_ratio=0.25, flipped=True, random_offset=False):
    #     """
    #     1) block_mask a contiguous chunk
    #     2) on the *remaining unmasked patches*, apply random_masking 
    #        at 'random_ratio' to further mask a subset.
        
    #     This merges large contiguous missing + some scattered missing patches.
    #     """
    #     # Step 1: block_masking
    #     x_block_masked, mask_block, ids_restore_block = self.block_masking(
    #         x, block_ratio=block_ratio, flipped=flipped, random_offset=random_offset
    #     )
    #     # x_block_masked shape: [N, len_keep_block, D]

    #     # The 'mask_block' is shape [N, L], but the unmasked portion was physically removed in x_block_masked.
    #     # We next apply random masking *only to the unmasked portion* in x_block_masked.

    #     N, L_block, D = x_block_masked.shape  
    #     # L_block = number_of_unmasked_patches_after_block

    #     if L_block == 0:
    #         # if block_ratio is too large, everything is masked - skip random masking
    #         return x_block_masked, mask_block, ids_restore_block

    #     # We'll do random masking on x_block_masked with ratio = random_ratio:
    #     # But we interpret random_ratio as fraction of patches among L_block
    #     # If random_ratio=0.25, we mask 25% of the already unmasked portion.
    #     len_keep_additional = int(L_block * (1 - random_ratio))

    #     noise = torch.rand(N, L_block, device=x_block_masked.device)
    #     ids_shuffle = torch.argsort(noise, dim=1)
    #     ids_restore_local = torch.argsort(ids_shuffle, dim=1)
    #     ids_keep_local = ids_shuffle[:, :len_keep_additional]

    #     x_final_masked = torch.gather(
    #         x_block_masked, dim=1,
    #         index=ids_keep_local.unsqueeze(-1).expand(-1, -1, D)
    #     )

    #     # We need a new mask that merges block_mask + random_mask
    #     # The 'mask_block' is full dimension [N, L], but we now further masked an additional chunk among the unmasked portion.
    #     # Let's define a "random mask" in that local sense:
    #     rand_mask_local = torch.ones(N, L_block, device=x.device)
    #     rand_mask_local[:, :len_keep_additional] = 0
    #     rand_mask_local = torch.gather(rand_mask_local, dim=1, index=ids_restore_local)

    #     # Now we have to place this "rand_mask_local" back into the global mask dimension
    #     # The global mask dimension = L (the original # patches).
    #     # But we only have indexing for the unmasked portion from block_mask. 
    #     # The block_mask function gave us 'ids_restore_block' to re-inject tokens into the full dimension.

    #     # We'll create a new mask array:
    #     combined_mask = mask_block.clone()  # start from block_mask result
    #     # combined_mask shape: [N, L], already has 1=masked, 0=unmasked from block_mask
    #     # The unmasked portion from block_mask is where combined_mask=0.

    #     # We need to figure out which patches in the original ordering were unmasked by block_mask. 
    #     # The indices of unmasked patches are those that appear in 'ids_shuffle_block[:len_keep_block]'. 
    #     # 'ids_restore_block' was for re-injecting them. Let's reconstruct them carefully:

    #     # Option: We can iterate through the block_mask again or we can do a direct approach:
    #     # Let's define an array "block_unmasked_positions" in the original ordering. 
    #     block_unmasked_positions = (combined_mask == 0)  # shape [N, L], boolean
    #     # Now for each sample, the #unmasked positions = L_block.

    #     # We apply the random mask to those unmasked positions:
    #     idx_rand_mask = 0
    #     for i in range(N):
    #         # block_unmasked_indices is the set of indices where combined_mask[i] = 0
    #         unmasked_indices = torch.where(block_unmasked_positions[i] == True)[0]  # shape [L_block]
    #         # Now 'rand_mask_local[i]' has shape [L_block], 1= newly masked, 0=unmasked
    #         combined_mask[i, unmasked_indices] = rand_mask_local[i]

    #     # So combined_mask is now block_mask OR random_mask. 
    #     # Then the final x_final_masked is shape [N, (len_keep_block * (1-random_ratio)), D].

    #     # We need a final 'ids_restore' for the combined approach. We'll do a similar "shuffle" logic:
    #     unmasked_indices_global = []
    #     for i in range(N):
    #         # where combined_mask[i,:] == 0
    #         idx_unmasked = torch.where(combined_mask[i] == 0)[0]  # shape [?,]
    #         unmasked_indices_global.append(idx_unmasked)
    #     # We'll have to unify this as a single ordering. But to keep it consistent with the single index approach, let's do the row-major approach again:

    #     # We'll do row-major re-ordering for the final gather:
    #     # For simplicity, let's just do the approach of building a "mask ordering" exactly like block_masking:
    #     # Because combined_mask is correct. We'll gather unmasked first, masked last:
    #     final_x = []
    #     final_ids_restore_list = []
    #     for i in range(N):
    #         unmasked_i = torch.where(combined_mask[i]==0)[0]
    #         masked_i   = torch.where(combined_mask[i]==1)[0]
    #         ordering_i = torch.cat([unmasked_i, masked_i], dim=0)  # shape [L]
    #         # gather them from x
    #         # But we need the original x, not x_block_masked or x_final_masked. 
    #         # Because x_final_masked is missing patches from the original dimension. 
    #         # Let's gather from x directly:

    #         x_masked_final_i = torch.index_select(x[i], dim=0, index=unmasked_i)
    #         # final gather shape [len_keep_total, D]
    #         final_x.append(x_masked_final_i)

    #         # create ids_restore
    #         ids_restore_i = torch.argsort(ordering_i)
    #         final_ids_restore_list.append(ids_restore_i.unsqueeze(0))

    #     final_x_masked = torch.nn.utils.rnn.pad_sequence(final_x, batch_first=True, padding_value=0.0)
    #     # The above might cause dimension mismatch if the keep length differs across samples. If all samples have the same mask ratio, it's consistent.

    #     final_ids_restore = torch.cat(final_ids_restore_list, dim=0)

    #     # final_x_masked shape is [N, len_keep_total, D]
    #     return final_x_masked, combined_mask, final_ids_restore


    def forward_encoder_dynamic(self, x, mask_mode='random',
                                mask_ratio=0.75, block_ratio=0.5, block_ratio_w=0.5, random_ratio=0.25,
                                flipped=True, random_offset=False):
        """
        A dynamic encoder that chooses a masking strategy based on mask_mode.

        mask_mode in ['random', 'block', 'block_bbox', 'combined'].

        - random: random_masking
        - block: block_masking
        - block_bbox: block_masking_bbox_bottom
        - combined: combined_masking

        The chosen masking strategy modifies x to x_masked and returns mask and ids_restore.
        Then passes x_masked through encoder (frozen + new blocks).

        Args:
            x: [N, 3, H, W] input images
            mask_mode (str): chosen masking mode
            mask_ratio (float): used if random mode
            block_ratio (float): fraction for block-based methods
            block_ratio_w (float): horizontal fraction if needed (block_bbox mode)
            random_ratio (float): fraction for combined random portion
            flipped (bool): if True, treat row=0 as bottom
            random_offset (bool): if True, random vertical offset in block modes.

        Returns:
            x_masked, mask, ids_restore
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        if mask_mode == 'random':
            x_masked, mask, ids_restore = self.random_masking(x, mask_ratio=mask_ratio)
        elif mask_mode == 'block':
            x_masked, mask, ids_restore = self.block_masking(
                x, block_ratio=block_ratio, flipped=flipped, random_offset=random_offset
            )
        elif mask_mode == 'block_bbox':
            x_masked, mask, ids_restore = self.block_masking_bbox_bottom(
                x, block_ratio=block_ratio, horizontal_random_offset=True, flipped=flipped
            )
        elif mask_mode == 'combined':
            x_masked, mask, ids_restore = self.combined_masking(
                x, block_ratio=block_ratio, random_ratio=random_ratio,
                flipped=flipped, horizontal_random_offset=True, random_offset=random_offset
            )
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # Pass through frozen encoder
        for blk in self.blocks:
            x_masked = blk(x_masked)

        # Pass through new trainable blocks
        for blk in self.new_blocks:
            x_masked = blk(x_masked)

        x_masked = self.norm(x_masked)
        return x_masked, mask, ids_restore


    # def forward_encoder(self, x, mask_ratio):
    #     """
    #     Forward pass through encoder.

    #     Args:
    #         x: Input images [N, 3, H, W]
    #         mask_ratio: fraction of patches to mask

    #     Returns:
    #         x_masked: latent representation after encoding
    #         mask: binary mask of which patches are masked
    #         ids_restore: indices for restoring patch order
    #     """
    #     x = self.patch_embed(x)
    #     x = x + self.pos_embed[:, 1:, :]  # Add positional embeddings (skip cls token)

    #     x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)

    #     cls_token = self.cls_token + self.pos_embed[:, :1, :]
    #     cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
    #     x_masked = torch.cat((cls_tokens, x_masked), dim=1)

    #     # Pretrained frozen encoder blocks
    #     for blk in self.blocks:
    #         x_masked = blk(x_masked)

    #     # New trainable blocks
    #     for blk in self.new_blocks:
    #         x_masked = blk(x_masked)

    #     x_masked = self.norm(x_masked)
    #     return x_masked, mask, ids_restore


    def forward_decoder(self, x, ids_restore):
        """
        Forward pass through the decoder.

        Args:
            x: Latent representation after encoding.
            ids_restore: Indices to restore original patch order.

        Returns:
            Reconstructed patches [N, L, patch_size**2 * in_chans].
        """
        # Embed latent tokens for decoding
        x = self.decoder_embed(x)

        # Add mask tokens to the sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x_ = torch.cat([x[:, :1, :], x_], dim=1)  # Reattach class token

        # Add positional embeddings
        x = x_ + self.decoder_pos_embed

        # Apply Transformer decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Project decoded tokens back to image space
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove class token
        return x


    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss using a hybrid L1+L2 approach.

        Args:
            imgs: original images [N, 3, H, W]
            pred: predicted patches [N, L, patch_size^2 * 3]
            mask: binary mask [N, L], 1 means masked patch.

        Returns:
            loss: scalar loss value
        """
        target = self.patchify(imgs)
        alpha = 0.5
        loss_l2 = (pred - target)**2
        loss_l1 = torch.abs(pred - target)
        loss_combined = alpha * loss_l1 + (1 - alpha) * loss_l2
        loss_per_patch = loss_combined.mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum()
        return loss

    # def forward_loss(self, imgs, pred, mask):
    #     """
    #     Compute reconstruction loss.

    #     Args:
    #         imgs: original images [N, 3, H, W]
    #         pred: predicted patches [N, L, patch_size^2 * 3]
    #         mask: binary mask [N, L], 1 means masked, 0 means unmasked
    #     """
    #     target = self.patchify(imgs)
    #     loss = (pred - target) ** 2
    #     loss = loss.mean(dim=-1)  # mean loss per patch

    #     # Compute loss only on masked patches
    #     loss = (loss * mask).sum() / mask.sum()
    #     return loss
    
    
    def forward(self, imgs,
                mask_mode='block_bbox',
                mask_ratio=0.75,
                block_ratio=0.5,
                block_ratio_w=0.5,
                random_ratio=0.25,
                flipped=True,
                random_offset=False):
        """
        Full forward pass:
        1) Apply chosen masking strategy.
        2) Encode masked input.
        3) Decode to reconstruct masked patches.
        4) Compute loss.

        Args:
            imgs: [N, 3, H, W]
            mask_mode: 'random', 'block', 'block_bbox', 'combined'
            mask_ratio: only relevant for random mode
            block_ratio: fraction for block-based methods
            block_ratio_w: horizontal fraction if needed
            random_ratio: fraction for random portion in combined mode
            flipped: if True, row=0 is bottom visually
            random_offset: if True, random vertical offset in block modes

        Returns:
            loss, pred, mask
        """
        latent, mask, ids_restore = self.forward_encoder_dynamic(
            x=imgs,
            mask_mode=mask_mode,
            mask_ratio=mask_ratio,
            block_ratio=block_ratio,
            block_ratio_w=block_ratio_w,
            random_ratio=random_ratio,
            flipped=flipped,
            random_offset=random_offset
        )
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    # def forward(self, imgs, mask_ratio=0.75):
    #     """
    #     Forward pass of the MAE:

    #     Args:
    #         imgs: [N, 3, H, W] input images
    #         mask_ratio: fraction of patches to mask

    #     Returns:
    #         loss: reconstruction loss
    #         pred: reconstructed patches
    #         mask: which patches were masked
    #     """
    #     latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
    #     pred = self.forward_decoder(latent, ids_restore)
    #     loss = self.forward_loss(imgs, pred, mask)
    #     return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

# decoder_depth changed from 8
def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=12, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks