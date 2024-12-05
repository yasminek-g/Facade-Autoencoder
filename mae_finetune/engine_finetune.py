# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import matplotlib.pyplot as plt
import os
import numpy as np

import torch

from timm.data import Mixup
from timm.utils import accuracy

import torch
from torch.cuda.amp import GradScaler, autocast
from util.misc import MetricLogger

import util.misc as misc
import util.lr_sched as lr_sched

import wandb

def save_visualizations(samples, predictions, epoch, output_dir, num_visuals=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Save input images and reconstructed images side by side for visualization.
    Includes debugging output for tensor ranges.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move tensors to CPU and convert to numpy
    samples = samples.cpu().numpy()
    predictions = predictions.cpu().detach().numpy()

    # Denormalize images if mean and std are provided
    if mean and std:
        mean = np.array(mean).reshape(1, 3, 1, 1)  # Reshape for broadcasting
        std = np.array(std).reshape(1, 3, 1, 1)
        samples = (samples * std) + mean
        predictions = (predictions * std) + mean

    # Clip values to [0, 1] range after denormalization
    samples = np.clip(samples, 0, 1)
    predictions = np.clip(predictions, 0, 1)

    samples = np.flip(samples, axis=2)  # Flip along the height (H)
    predictions = np.flip(predictions, axis=2)  # Flip along the height (H)

    for i in range(min(len(samples), num_visuals)):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Original Image
        axes[0].imshow(samples[i].transpose(1, 2, 0))  # Convert (C, H, W) -> (H, W, C)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        # Reconstructed Image
        axes[1].imshow(predictions[i].transpose(1, 2, 0))
        axes[1].set_title("Reconstructed Image")
        axes[1].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"epoch_{epoch}_sample_{i}.png")
        plt.savefig(save_path)
        plt.close(fig)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler,
                    max_norm=0, mixup_fn=None, log_writer=None, args=None):
    """
    Trains the model for one epoch.

    Args:
        model: The model being trained.
        criterion: The loss function (can be None as loss is computed inside the model).
        data_loader: Training data loader.
        optimizer: Optimizer used for training.
        device: The device to run the model on.
        epoch: Current epoch number.
        loss_scaler: Scaler for mixed precision training.
        max_norm: Max gradient norm for clipping.
        mixup_fn: Optional Mixup function.
        log_writer: Writer for logging to TensorBoard.
        args: Additional arguments (e.g., batch size, accumulation steps).

    Returns:
        A dictionary with training statistics.
    """
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    scaler = GradScaler()  # Mixed precision training scaler
    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1

    optimizer.zero_grad()

    for data_iter_step, (samples, informative_mask) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)
        informative_mask = informative_mask.to(device, non_blocking=True)

        # Forward pass with mixed precision
        with autocast():
            loss, pred, mask = model(samples, mask_ratio=args.mask_ratio, informative_mask=informative_mask)
            loss = loss / accum_iter  # Normalize loss for gradient accumulation

        # Backward pass
        scaler.scale(loss).backward()

        if (data_iter_step + 1) % accum_iter == 0:  # Update gradients after `accum_iter` steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        # Optional logging to TensorBoard or Weights & Biases
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_writer.add_scalar("train_loss", loss_value, epoch * len(data_loader) + data_iter_step)
            log_writer.add_scalar("lr", lr, epoch * len(data_loader) + data_iter_step)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, output_dir, num_visuals=5):
    """
    Evaluate the model on validation data and visualize results.

    Args:
        data_loader (DataLoader): Validation data loader.
        model (nn.Module): The model being evaluated.
        device (torch.device): Device to run evaluation.
        epoch (int): Current epoch (for visualization filenames).
        output_dir (str): Directory to save visualizations.
        num_visuals (int): Number of visualizations to save.
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # Directory for visualizations
    vis_dir = os.path.join(output_dir, "val_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for batch_idx, (images, informative_mask) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        informative_mask = informative_mask.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            loss, pred, mask = model(images, mask_ratio=0.75, informative_mask=informative_mask)

        # Visualize a few samples
        if batch_idx < num_visuals:
            reconstructions = model.unpatchify(pred)
            save_visualizations(
                images.cpu(), reconstructions.cpu(), epoch,
                vis_dir, num_visuals=num_visuals
            )

    # Synchronize metrics across GPUs
    metric_logger.synchronize_between_processes()
    print("Validation Stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}