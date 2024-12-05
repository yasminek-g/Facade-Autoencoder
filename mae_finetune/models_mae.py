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
    """Masked Autoencoder with VisionTransformer backbone."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # Encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)

        # Pretrained encoder blocks (to be frozen)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])

        # New trainable blocks
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
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

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
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def freeze_pretrained_weights(self):
        """Freeze the weights of the pretrained encoder."""
        for param in self.blocks.parameters():
            param.requires_grad = False
        print("Pretrained encoder blocks have been frozen.")

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
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
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, informative_mask):
        """
        Randomly masks patches, ensuring padding regions (via informative_mask) are always masked.

        Args:
            x: Input features of shape [N, L, D].
            mask_ratio: Ratio of patches to mask.
            informative_mask: Binary mask of shape [N, L] where 1 indicates informative regions.

        Returns:
            x_masked: Masked input features.
            mask: Combined masking (random + padding).
            ids_restore: Indices to restore original order.
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        # Apply random noise only to informative patches
        noise = torch.rand(N, L, device=x.device) * informative_mask
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, informative_mask):
        """
        Forward pass through encoder, including new trainable blocks.

        Args:
            x: Input image patches [N, L, D].
            mask_ratio: Ratio of patches to mask.
            informative_mask: Mask to identify valid regions [N, L].

        Returns:
            x_masked: Latent representation after encoding.
            mask: Mask used during encoding.
            ids_restore: Indices to restore original patch order.
        """
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]  # Add positional embeddings (skip cls token)

        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio, informative_mask)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # Pretrained frozen encoder blocks
        for blk in self.blocks:
            x_masked = blk(x_masked)

        # New trainable blocks
        for blk in self.new_blocks:
            x_masked = blk(x_masked)

        x_masked = self.norm(x_masked)
        return x_masked, mask, ids_restore


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


    def forward_loss(self, imgs, pred, mask, informative_mask):
        """
        Compute reconstruction loss, ignoring padding.

        Args:
            imgs: Original images [N, 3, H, W].
            pred: Predicted patches [N, L, patch_size**2 * in_chans].
            mask: Mask array [N, L].
            informative_mask: Binary mask identifying valid regions [N, L].

        Returns:
            Loss value.
        """
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Average over patch dimensions

        combined_mask = mask * informative_mask
        loss = (loss * combined_mask).sum() / combined_mask.sum()
        return loss


    def forward(self, imgs, mask_ratio=0.75, informative_mask=None):
        """
        Forward pass for the MAE model with new trainable blocks.

        Args:
            imgs: Input images [N, 3, H, W].
            mask_ratio: Ratio of patches to mask.
            informative_mask: Binary mask identifying valid regions [N, L].

        Returns:
            loss: Reconstruction loss.
            pred: Reconstructed patches.
            mask: Mask used during reconstruction.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, informative_mask)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask, informative_mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
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