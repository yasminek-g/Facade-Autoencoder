import random
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import os
from PIL import Image

def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

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

# function to combine my original and reconstructed images
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