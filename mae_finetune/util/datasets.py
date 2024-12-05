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
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import shutil
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
    transform = build_transform(is_train, args)

    if is_train:
        image_dir = os.path.join(args.data_path, '/home/kroknes/complete_processed_npy/train/images_npy')
        mask_dir = os.path.join(args.data_path, '/home/kroknes/complete_processed_npy/train/masks_npy')
    else:
        image_dir = os.path.join(args.data_path, '/home/kroknes/complete_processed_npy/val/images_npy')
        mask_dir = os.path.join(args.data_path, '/home/kroknes/complete_processed_npy/val/masks_npy')

    dataset = NumpyDataset(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

    print(f"Dataset size: {len(dataset)} ({'training' if is_train else 'validation'})")
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

    transform = transforms.Compose([
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


class NumpyDataset(Dataset):
    """
    Custom dataset class for loading images and padding masks.

    Args:
        image_dir (str): Directory containing image .npy files.
        mask_dir (str): Directory containing mask .npy files.
        transform (callable, optional): Transformations to apply to the images.
        patch_size (int): Patch size for the MAE model.
    """
    def __init__(self, image_dir, mask_dir, transform=None, patch_size=16):
        self.image_paths = sorted([
            os.path.join(image_dir, fname) for fname in os.listdir(image_dir)
            if fname.endswith('resized_padded.npy')
        ])
        self.mask_paths = sorted([
            os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)
            if fname.endswith('mask.npy')
        ])

        assert len(self.image_paths) > 0, "No image files found in the specified directory."
        assert len(self.mask_paths) > 0, "No mask files found in the specified directory."

        self.image_paths, self.mask_paths = self._match_files(self.image_paths, self.mask_paths)
        assert len(self.image_paths) > 0, "No matching image and mask files found."

        self.transform = transform
        self.patch_size = patch_size

    def _match_files(self, image_paths, mask_paths):
        """
        Matches image files with corresponding mask files.

        Args:
            image_paths (list): List of image file paths.
            mask_paths (list): List of mask file paths.

        Returns:
            tuple: Matched image and mask file paths.
        """
        def extract_id(fname, pattern):
            start = fname.find('1_') + 2
            if pattern == 'image':
                end = fname.find('__full_rgb__resized_padded')
            elif pattern == 'mask':
                end = fname.find('__full_rgb__mask')
            return fname[start:end]

        image_dict = {extract_id(os.path.basename(p), 'image'): p for p in image_paths}
        mask_dict = {extract_id(os.path.basename(p), 'mask'): p for p in mask_paths}

        common_ids = set(image_dict.keys()) & set(mask_dict.keys())
        matched_image_paths = [image_dict[id_] for id_ in sorted(common_ids)]
        matched_mask_paths = [mask_dict[id_] for id_ in sorted(common_ids)]

        return matched_image_paths, matched_mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: Image tensor and informative mask tensor.
        """
        # Load image and mask
        image = np.load(self.image_paths[idx]).astype(np.float32) / 255.0  # [H, W, C] scaled to [0, 1]
        informative_mask = np.load(self.mask_paths[idx]).astype(np.float32)  # [H, W]

        # Convert image to tensor and apply transformations
        image_tensor = torch.from_numpy(image).float().permute(2, 0, 1)  # [C, H, W]
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Resize mask to match patch size
        p = self.patch_size
        num_patches = image_tensor.shape[1] // p
        informative_mask_tensor = torch.from_numpy(informative_mask).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        informative_mask_tensor = torch.nn.functional.interpolate(
            informative_mask_tensor, size=(num_patches, num_patches), mode='nearest'
        ).squeeze(0).squeeze(0).view(-1)  # [L]

        return image_tensor, informative_mask_tensor