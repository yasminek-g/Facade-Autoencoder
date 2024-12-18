# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


def build_dataset(is_train, args):
    """
    Builds the dataset for training or validation.

    Args:
        is_train (bool): Whether to build the training dataset.
        args: Command-line arguments containing data_path, input_size, etc.

    Returns:
        Dataset: An instance of NumpyDataset.
    """
    full_npy_dir = os.path.join(args.data_path, 'complete_npy_224')
    
    # Load all .npy image files
    all_image_paths = sorted([
        os.path.join(full_npy_dir, fname) for fname in os.listdir(full_npy_dir)
        if fname.endswith('.npy')
    ])
    
    assert len(all_image_paths) > 0, "No .npy image files found in the specified directory."
    
    # Split into train and val sets
    train_paths, val_paths = train_test_split(all_image_paths, test_size=0.2, random_state=42)

    transform = build_transform(is_train, args)
    if is_train:
        dataset = NumpyDataset(train_paths, transform=transform)
        print(f"Dataset size: {len(dataset)} (training)")
    else:
        dataset = NumpyDataset(val_paths, transform=transform)
        print(f"Dataset size: {len(dataset)} (validation)")
    
    return dataset


def build_transform(is_train, args):
    """
    Builds the transformation pipeline for the dataset.

    Args:
        is_train (bool): Whether the transformations are for training.
        args: Command-line arguments containing dataset parameters.

    Returns:
        Transform: A torchvision transform.
    """
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std

    transform_list = [
        transforms.ToPILImage(),
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    transform = transforms.Compose(transform_list)
    return transform


class NumpyDataset(Dataset):
    """
    Custom dataset class for loading images from .npy files.

    Args:
        image_paths (list): List of image file paths.
        transform (callable, optional): Transformations to apply to the images.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = np.load(self.image_paths[idx]).astype(np.float32) / 255.0  # [H, W, C]

        # Apply transform
        # Since we're now using a transform that expects a PIL image, we need to be sure
        # the transform pipeline includes a ToPILImage() step.
        if self.transform:
            # image: numpy array (H, W, C)
            # transform pipeline includes ToPILImage so it can handle numpy arrays with shape (H,W,C)
            image = self.transform(image)  # will become a Tensor [C, H, W]

        return image