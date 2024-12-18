import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
import os
from PIL import Image

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    #print(f"sequences shape: {sequences.shape}, indexes shape: {indexes.shape}")
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

# Block-based masking for patches
class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        """
        Block-based masking for patches.

        Args:
        - ratio (float): Fraction of the image to be masked.
        """
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        """
        Applies block masking to the patches.

        Args:
        - patches (torch.Tensor): Input patches of shape (T, B, C).

        Returns:
        - masked_patches (torch.Tensor): Patches after block masking.
        - forward_indexes (torch.Tensor): Mapping of visible patches to the original sequence.
        - backward_indexes (torch.Tensor): Reverse mapping to reconstruct the full sequence.
        """
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))  # Number of patches to keep.

        # Calculate block size based on T (total patches)
        num_patches_per_side = int(T ** 0.5)
        assert T == num_patches_per_side ** 2, "Patches must form a square grid."

        indexes = [self.block_indexes(num_patches_per_side, remain_T) for _ in range(B)]
        forward_indexes = torch.as_tensor(
            np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)
        backward_indexes = torch.as_tensor(
            np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long
        ).to(patches.device)

        # Shuffle and mask patches
        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes

    def block_indexes(self, num_patches_per_side, remain_T):
        """
        Generate forward and backward indexes for block masking.

        Args:
        - num_patches_per_side (int): Number of patches per side of the grid.
        - remain_T (int): Number of patches to keep.

        Returns:
        - forward_indexes (np.ndarray): Indexes of kept patches after masking.
        - backward_indexes (np.ndarray): Indexes for reconstructing the full sequence.
        """
        grid_size = num_patches_per_side ** 2  # Total patches in the grid.
        mask_T = grid_size - remain_T  # Number of patches to mask.
        block_side = int(mask_T ** 0.5)  # Block size in terms of patches.

        # Randomly select the top-left corner of the block
        max_i = num_patches_per_side - block_side
        max_j = num_patches_per_side - block_side
        i = np.random.randint(0, max_i + 1)
        j = np.random.randint(0, max_j + 1)

        # Generate a 2D grid mask
        mask = np.zeros((num_patches_per_side, num_patches_per_side), dtype=np.float32)
        mask[i:i + block_side, j:j + block_side] = 1  # Mask the block.

        # Flatten the mask
        mask = mask.flatten()

        # Determine forward and backward indexes
        forward_indexes = np.where(mask == 0)[0]  # Keep unmasked patches.
        backward_indexes = np.argsort(np.concatenate((forward_indexes, np.where(mask == 1)[0])))

        return forward_indexes, backward_indexes


# Encoder for Masked Autoencoders
class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


# Decoder for Masked Autoencoders
class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask


# Complete MAE Vision Transformer
class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.5
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask

# losses used during training
class MaskedMSELoss(nn.Module):
    def __init__(self):

        super(MaskedMSELoss, self).__init__()

    def forward(self, predicted, target, mask):
        """
        Compute MSE loss on masked patches only.

        Parameters:
        - predicted: Reconstructed image tensor (B, C, H, W)
        - target: Original image tensor (B, C, H, W)
        - mask: Binary mask indicating masked regions

        Returns:
        - loss: Mean squared error loss over masked regions
        """
        masked_loss = ((predicted - target) ** 2 * mask).sum()
        num_masked_patches = mask.sum()
        loss = masked_loss / (num_masked_patches + 1e-8)  # Avoid division by zero
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, layers=[2, 7, 12], requires_grad=False, device='cuda'):
        """
        Extract features from VGG19 layers to compute perceptual loss.
        - layers: Intermediate VGG layers for feature extraction.
        - device: Device where VGG model should be moved (default: 'cuda')
        """
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features.eval()  # Load pre-trained VGG
        self.layers = layers
        self.vgg = vgg[:max(layers) + 1].to(device)  # Move VGG to the correct device
        for param in self.vgg.parameters():
            param.requires_grad = requires_grad

        self.device = device

    def forward(self, generated, target, mask):
        """
        Compute perceptual loss for masked patches only.

        Parameters:
        - generated: Reconstructed image tensor (B, C, H, W)
        - target: Original image tensor (B, C, H, W)
        - mask: Binary mask indicating masked regions
        """
        # Move inputs to the same device as the VGG model
        generated = generated.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)

        # Apply mask
        generated_masked = generated * mask
        target_masked = target * mask

        # Feature extraction
        loss = 0.0
        for layer in self.layers:
            generated_features = self.vgg[:layer](generated_masked)
            target_features = self.vgg[:layer](target_masked)
            loss += F.mse_loss(generated_features, target_features)

        return loss

# Loading numpy image dataset (numpy images created via the facade_extraction.py file)

class NumpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom Dataset to load images from .npy files.
        Args:
            root_dir (str): Directory containing .npy image files.
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load the .npy file
        img_path = self.image_files[idx]
        image = np.load(img_path)  # Loaded as a NumPy array, likely (H, W, C)

        # Convert to tensor and ensure the shape is (C, H, W)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Rearrange channels

        # Apply any transforms (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        return image


def combine_images_side_by_side(original, reconstructed):
    """
    Combines the original and reconstructed images side by side.

    Args:
        original: Tensor of original images, shape (C, H, W).
        reconstructed: Tensor of reconstructed images, shape (C, H, W).

    Returns:
        A combined numpy array with original and reconstructed side by side.
    """
    original = original.cpu().permute(1, 2, 0).numpy()  # Convert to HWC format
    reconstructed = reconstructed.cpu().permute(1, 2, 0).numpy()

    # Unnormalize to [0, 1] range
    original = np.clip(original * 0.5 + 0.5, 0, 1)
    reconstructed = np.clip(reconstructed * 0.5 + 0.5, 0, 1)

    # Concatenate the images horizontally
    combined = np.concatenate((original, reconstructed), axis=1)
    return combined





