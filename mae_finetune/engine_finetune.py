# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import matplotlib.pyplot as plt
import os
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast

from timm.data import Mixup
from timm.utils import accuracy

from util.misc import MetricLogger
import util.misc as misc
import torch.nn.functional as F

import wandb

from torchmetrics import StructuralSimilarityIndexMeasure
from lpips import LPIPS

def save_visualizations(samples, predictions, epoch, output_dir, num_visuals=5, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Save input images and reconstructed images side by side for visualization.

    Args:
        samples (torch.Tensor): Original input images [N, 3, H, W].
        predictions (torch.Tensor): Reconstructed images [N, 3, H, W].
        epoch (int): Current epoch number for filename tagging.
        output_dir (str): Directory to save visualization images.
        num_visuals (int): Number of samples to visualize.
        mean (list): Mean values for denormalization.
        std (list): Std values for denormalization.

    This function:
      - Denormalizes the samples and predictions.
      - Optionally flips images vertically (as per code).
      - Saves side-by-side comparisons of original and reconstructed images to the output directory.

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
                    max_norm=0, mixup_fn=None, log_writer=None, args=None,
                    perceptual_model=None, d_model=None, d_optimizer=None,
                    refinement_net=None, refinement_optimizer=None,
                    # Dynamic masking arguments
                    mask_mode='random',
                    block_ratio=0.5,
                    random_ratio=0.25,
                    flipped=True,
                    random_offset=False,
                    block_ratio_w=0.5):
    """
    Train the model for one epoch, possibly using perceptual loss, GAN loss, and refinement net.

    Args:
        model (nn.Module): The MAE model being trained.
        criterion: The loss function (can be None, since the model computes loss internally).
        data_loader (DataLoader): Training data loader.
        optimizer (Optimizer): Optimizer used for training.
        device (torch.device): Device to run training on.
        epoch (int): Current epoch number.
        loss_scaler: Scaler for mixed precision.
        max_norm (float): Max gradient norm for clipping.
        mixup_fn: Optional Mixup function (not used if mixup is not required).
        log_writer: SummaryWriter for TensorBoard logging (if any).
        args (argparse.Namespace): Additional training args (e.g. batch size).
        perceptual_model: A model for perceptual loss if use_perceptual_loss is True.
        d_model: Discriminator model if use_gan_loss is True.
        d_optimizer: Optimizer for the discriminator if use_gan_loss is True.
        refinement_net: Refinement U-Net if use_refinement is True.
        refinement_optimizer: Optimizer for the refinement net if use_refinement is True.

        mask_mode (str): Masking mode: 'random', 'block', 'block_bbox', or 'combined'.
        block_ratio (float): Fraction of patches to mask with a block (if using block-based methods).
        random_ratio (float): Fraction of patches to mask randomly (if combined).
        flipped (bool): If True, row=0 is bottom visually.
        random_offset (bool): If True, random vertical offset for block mask.
        block_ratio_w (float): Horizontal fraction for the block width in block_bbox mode.

    Returns:
        dict: Training statistics (averaged losses, etc.)

    Note:
    - This function performs a forward pass with mixed precision.
    - If perceptual loss is requested, it is added to the MAE loss.
    - If GAN loss is requested, it trains the discriminator first, then updates the generator (MAE).
    - If refinement net is used, it adds a refinement step after the MAE forward pass.
    - gradient clipping is done with torch.nn.utils.clip_grad_norm_.
    - metrics (MSE, SSIM, LPIPS) are computed and logged to W&B.
    - If log_writer is provided, also logs to TensorBoard.
    
    Flagging Unused:
    - `mixup_fn` is passed but not used in this code snippet.
    - `criterion` is passed but not used since the model computes the loss internally.
    - If args.use_gan_loss=False, then d_model and d_optimizer remain unused.
    - If args.use_refinement=False, refinement_net and refinement_optimizer remain unused.
    """
    model.train(True)
    if refinement_net is not None:
        refinement_net.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = 10

    scaler = GradScaler()  # Mixed precision training scaler
    accum_iter = args.accum_iter if hasattr(args, 'accum_iter') else 1

    optimizer.zero_grad()
    if d_optimizer is not None:
        d_optimizer.zero_grad()
    if refinement_optimizer is not None:
        refinement_optimizer.zero_grad()

    # Initialize SSIM and LPIPS
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LPIPS(net='vgg').to(device)

    for data_iter_step, samples in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        samples = samples.to(device, non_blocking=True)  # On GPU0

        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            # Call the model with dynamic masking arguments
            loss, pred, mask = model(
                samples,
                mask_mode=mask_mode,
                mask_ratio=args.mask_ratio,
                block_ratio=block_ratio,
                block_ratio_w=block_ratio_w,
                random_ratio=random_ratio,
                flipped=flipped,
                random_offset=random_offset
            )
            recon = model.unpatchify(pred)
            recon = recon.to(device) # Reconstruction on GPU0

        if args.use_perceptual_loss and perceptual_model is not None:
            perc_loss = perceptual_model(recon.float(), samples.float())
            loss = loss + perc_loss

        if args.use_gan_loss and d_model is not None and d_optimizer is not None:
            # Move reconstruction to GPU1 for discriminator
            recon_on_gpu1 = recon.to('cuda:1', non_blocking=True)
            samples_on_gpu1 = samples.to('cuda:1', non_blocking=True)

            # Train discriminator
            d_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                real_pred = d_model(samples_on_gpu1)
                real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))

                fake_pred = d_model(recon_on_gpu1.detach())
                fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))

                d_loss = 0.5 * (real_loss + fake_loss)

            scaler.scale(d_loss).backward()
            scaler.step(d_optimizer)
            scaler.update()
            d_optimizer.zero_grad()

            # Now train the generator (MAE) with GAN loss
            optimizer.zero_grad()
            if refinement_net is not None and refinement_optimizer is not None:
                refinement_optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                fake_pred = d_model(recon_on_gpu1)
                gan_loss = F.mse_loss(fake_pred, torch.ones_like(fake_pred)) * args.gan_loss_weight

                gan_loss = gan_loss.to(device)      
                total_loss = loss + gan_loss
        else:
            total_loss = loss

        # If using refinement net, move reconstruction to refinement GPU and process
        if refinement_net is not None and refinement_optimizer is not None:
            refinement_device = next(refinement_net.parameters()).device
            recon_on_gpu_r = recon.to(refinement_device, non_blocking=True)
            samples_on_gpu_r = samples.to(refinement_device, non_blocking=True)

            with torch.cuda.amp.autocast():
                refined = refinement_net(recon_on_gpu_r)
                # Compute refinement loss on GPU1
                ref_loss = F.mse_loss(refined, samples_on_gpu_r)

                ref_loss = ref_loss.to(device)
                total_loss = total_loss + ref_loss

        total_loss = total_loss / accum_iter

        # Backward pass
        scaler.scale(total_loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip at norm 1.0

        if (data_iter_step + 1) % accum_iter == 0:  # Update gradients after `accum_iter` steps
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Step the optimizer for the refinement network only if refinement_loss was computed
            if refinement_net is not None and refinement_optimizer is not None:
                scaler.step(refinement_optimizer)
                scaler.update()
                refinement_optimizer.zero_grad()

        torch.cuda.synchronize()

        # Compute metrics
        mse_loss = F.mse_loss(recon, samples).item()
        ssim_score = ssim_metric(recon, samples).item()
        lpips_score = lpips_metric(recon, samples).mean().item()

        # Log metrics to wandb
        wandb.log({
            "train_loss": total_loss.item(),
            "train_MSE": mse_loss,
            "train_SSIM": ssim_score,
            "train_LPIPS": lpips_score,
            "epoch": epoch
        })

        loss_value = total_loss.item()
        metric_logger.update(loss=loss_value)

        # Optional logging to TensorBoard
        if log_writer and (data_iter_step + 1) % accum_iter == 0:
            lr = optimizer.param_groups[0]["lr"]
            log_writer.add_scalar("train_loss", loss_value, epoch * len(data_loader) + data_iter_step)
            log_writer.add_scalar("lr", lr, epoch * len(data_loader) + data_iter_step)

    # Gather stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, output_dir, num_visuals=5,
             # Dynamic masking approach if you want consistent validation
             mask_mode='block',
             mask_ratio=0.75,
             block_ratio=0.5,
             random_ratio=0.25,
             flipped=True,
             random_offset=False):
    """
    Evaluate the model on validation data and visualize results.

    Args:
        data_loader (DataLoader): Validation data loader.
        model (nn.Module): The model being evaluated.
        device (torch.device): Device to run evaluation on.
        epoch (int): Current epoch (for visualization filenames).
        output_dir (str): Directory to save visualizations.
        num_visuals (int): Number of visualizations to save.
        mask_mode (str): Masking mode to use during validation.
        mask_ratio (float): Ratio of patches to mask if mode='random'.
        block_ratio (float): Block ratio if using block-based methods.
        random_ratio (float): Random masking ratio if combined.
        flipped (bool): If True, row=0 is bottom visually.
        random_offset (bool): If True, random vertical offset for block masking.

    Returns:
        dict: Validation statistics (averaged MSE, SSIM, LPIPS).

    Note:
    - Runs forward pass with the given mask_mode and ratios.
    - Computes MSE, SSIM, and LPIPS.
    - Logs averaged results to wandb.
    - Saves visualizations to output_dir/val_visualizations/.
    """
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # Initialize SSIM and LPIPS metrics
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LPIPS(net='vgg').to(device)

    # Directory for visualizations
    vis_dir = os.path.join(output_dir, "val_visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Initialize accumulators for metrics
    total_mse_loss = 0
    total_ssim_score = 0
    total_lpips_score = 0
    num_batches = 0

    for batch_idx, images in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)

        # Forward pass
        with torch.cuda.amp.autocast():
            loss, pred, mask = model(
                images,
                mask_mode=mask_mode,
                mask_ratio=mask_ratio,
                block_ratio=block_ratio,
                random_ratio=random_ratio,
                flipped=flipped,
                random_offset=random_offset
            )
            reconstructions = model.unpatchify(pred)

        # Compute metrics
        mse_loss = F.mse_loss(reconstructions, images).item()
        ssim_score = ssim_metric(reconstructions, images).item()
        lpips_score = lpips_metric(reconstructions, images).mean().item()

        total_mse_loss += mse_loss
        total_ssim_score += ssim_score
        total_lpips_score += lpips_score
        num_batches += 1

        # Visualize a few samples
        if batch_idx < num_visuals:
            save_visualizations(
                images.cpu(), reconstructions.cpu(), epoch,
                vis_dir, num_visuals=num_visuals
            )

        # Compute average metrics
        avg_mse_loss = total_mse_loss / num_batches
        avg_ssim_score = total_ssim_score / num_batches
        avg_lpips_score = total_lpips_score / num_batches

        print(f"Epoch {epoch}: avg_mse_loss={avg_mse_loss}, avg_ssim_score={avg_ssim_score}, avg_lpips_score={avg_lpips_score}")
        # Log metrics to wandb once per epoch
        wandb.log({
            "val_MSE": avg_mse_loss,
            "val_SSIM": avg_ssim_score,
            "val_LPIPS": avg_lpips_score,
            "epoch": epoch
        }, commit=True)

    print(f"Epoch {epoch}: val_MSE={avg_mse_loss:.4f}, val_SSIM={avg_ssim_score:.4f}, val_LPIPS={avg_lpips_score:.4f}")

    # Synchronize metrics across GPUs
    metric_logger.synchronize_between_processes()
    print("Validation Stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}