# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
from functools import partial
import wandb

# Custom utilities and models
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_finetune import train_one_epoch, evaluate
from util.datasets import build_dataset
from util.lr_decay import param_groups_lrd


from torchmetrics import StructuralSimilarityIndexMeasure
from lpips import LPIPS

from util.perceptual_loss import PerceptualLoss
from util.discriminator import PatchDiscriminator
from util.refinement_net import RefinementNet

# Ensure timm version compatibility
assert timm.__version__ == "0.3.2", "Incompatible timm version. Please use version 0.3.2."

def get_args_parser():
    """
    Creates an argument parser for the MAE fine-tuning script.
    
    Returns:
        argparse.ArgumentParser: The argument parser with all configurations.
    """
    parser = argparse.ArgumentParser('MAE fine-tuning for reconstruction', add_help=False)

    # General training parameters
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio for MAE')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')

    # Paths and directories
    parser.add_argument('--finetune', default='/home/kroknes/mae_visualize_vit_large_ganloss.pth', type=str,
                        help='Path to the pretrained checkpoint')
    parser.add_argument('--output_dir', default='./output_dir', help='Directory to save outputs')
    parser.add_argument('--log_dir', default='./output_dir', help='Directory to save logs')
    parser.add_argument('--data_path', default='/home/kroknes/', type=str,
                        help='Dataset path')   

    # Device and performance settings
    parser.add_argument('--device', default='cuda', help='Device for training/testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory in DataLoader')

    # Resume training
    parser.add_argument('--resume', default='', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')

    # Learning rate schedule
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')

    # Loss functions
    parser.add_argument('--use_perceptual_loss', action='store_true', help='Use perceptual loss')
    parser.add_argument('--perceptual_loss_weight', type=float, default=0.1, help='Weight of perceptual loss')
    parser.add_argument('--use_gan_loss', action='store_true', 
                        help='Use GAN-based adversarial loss during training')
    parser.add_argument('--gan_loss_weight', type=float, default=0.01, 
                        help='Weight of the GAN generator loss term')

    # Refinement network
    parser.add_argument('--use_refinement', action='store_true', 
                        help='Use a refinement U-Net after reconstruction')
    parser.add_argument('--refinement_lr', type=float, default=1e-4, 
                        help='Learning rate for the refinement U-Net')

    return parser


def main(args):
    """
    Main training and evaluation loop for MAE-based image reconstruction.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Initialize WandB for experiment tracking
    wandb.init(project="mae-reconstruction", config=args)
    torch.cuda.empty_cache()

    device = torch.device('cuda:0')  # Main model and perceptual loss on GPU 0
    device_d = torch.device('cuda:1')  # Discriminator on GPU 1 (may remain unused if use_gan_loss=False)
    device_r = torch.device('cuda:1')  # Refinement net on GPU 1 (may remain unused if use_refinement=False)

    cudnn.benchmark = True

    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=True
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False
    )

    # Initialize MAE model
    model = models_mae.mae_vit_large_patch16_dec512d8b(
        img_size=args.input_size,
        in_chans=3,
        norm_pix_loss=False  # Ensure padding is ignored in loss
    )

    if args.finetune and not args.eval:
        torch.cuda.empty_cache()
        checkpoint_path = '/home/kroknes/mae_visualize_vit_large_ganloss.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cuda', mmap=True)

        print(f"Loading pretrained checkpoint from: {checkpoint_path}")
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # Filter out unnecessary keys and load the pretrained weights
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(f"Loaded pretrained weights with message: {msg}")

        # Freeze encoder layers
        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for blk in model.blocks:
            for param in blk.parameters():
                param.requires_grad = False

        print("Pretrained encoder layers frozen.")

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters / 1e6:.2f}M")

    # Create parameter groups with layer-wise lr decay
    param_groups = param_groups_lrd(model, weight_decay=args.weight_decay, no_weight_decay_list=model.no_weight_decay(), layer_decay=0.75)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    loss_scaler = NativeScaler()

    # Resume training if applicable
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from checkpoint: {args.resume}")

    # Conditionally instantiate perceptual_model if use_perceptual_loss is True
    perceptual_model = None
    if args.use_perceptual_loss:
        perceptual_model = PerceptualLoss(weight=args.perceptual_loss_weight, device=device)

    # Conditionally instantiate a discriminator for GAN loss if use_gan_loss is True
    if args.use_gan_loss:
        d_model = PatchDiscriminator().to(device_d)
        d_optimizer = torch.optim.AdamW(d_model.parameters(), lr=1e-4, betas=(0.5, 0.999))
    else:
        d_model = None
        d_optimizer = None

    # If using refinement:
    if args.use_refinement:
        refinement_net = RefinementNet().to(device_r)
        refinement_optimizer = torch.optim.AdamW(refinement_net.parameters(), lr=args.refinement_lr)
    else:
        refinement_net = None
        refinement_optimizer = None

    if refinement_net is not None:
        for param in refinement_net.parameters():
            param.requires_grad = True

    # Logging
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # -------------------------------------------------
        # Decide the masking mode for this epoch
        # -------------------------------------------------
        if epoch < args.warmup_epochs:
            current_mask_mode = 'random'   # Purely random masking
            current_block_ratio = 0.0      # Not used in 'random' mode, but keep consistent
            current_block_ratio_w = 0.0    # No horizontal block width yet
            current_random_ratio = 0.0     # Not used here
            current_flipped = True
            current_random_offset = False
        else:
            # After warmup, define progress from 0 to 1
            progress = (epoch - args.warmup_epochs) / float(args.epochs - args.warmup_epochs)
            progress = max(0, min(1, progress))

            # Total mask ratio is always 0.75
            # Start: block=0.5, random=0.25 (sum=0.75)
            # End:   block=0.75, random=0.0 (sum=0.75)

            start_block_ratio = 0.4
            end_block_ratio = 0.6
            current_block_ratio = start_block_ratio + progress * (end_block_ratio - start_block_ratio)
            # This grows from 0.5 to 0.75

            start_random_ratio = 0.35
            end_random_ratio = 0.15
            current_random_ratio = start_random_ratio + (1 - progress) * (end_random_ratio - start_random_ratio)
            # This decreases from 0.25 to 0.0 as progress goes to 1

            start_block_ratio_w = 0.5
            end_block_ratio_w = 0.75
            current_block_ratio_w = start_block_ratio_w + progress * (end_block_ratio_w - start_block_ratio_w)

            current_mask_mode = 'combined'
            current_flipped = True
            current_random_offset = True

        # Training Phase
        train_stats = train_one_epoch(
            model=model,
            criterion=None,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args,
            perceptual_model=perceptual_model,
            d_model=d_model,
            d_optimizer=d_optimizer,
            refinement_net=refinement_net,
            refinement_optimizer=refinement_optimizer,
            mask_mode=current_mask_mode,
            block_ratio=current_block_ratio,
            random_ratio=current_random_ratio,
            flipped=current_flipped,
            random_offset=current_random_offset,
            block_ratio_w=current_block_ratio_w
        )

        # Save Model Checkpoint
        if args.output_dir:
            final_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs - 1,
            }, final_checkpoint_path)
            print(f"Final checkpoint saved at {final_checkpoint_path}")

        # Step the scheduler
        scheduler.step()

        # Log Training Metrics
        wandb.log({
            "train_loss": train_stats["loss"],
            "lr": scheduler.get_last_lr()[0],
            "epoch": epoch,
        })

        # Validation Phase (Run after each epoch)
        print("Evaluating model on validation set...")
        model.eval()  # Set model to evaluation mode
        val_images = next(iter(data_loader_val))
        val_images = val_images.to(device, non_blocking=True)

        # Initialize SSIM and LPIPS metrics
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_metric = LPIPS(net='vgg').to(device)

        with torch.no_grad():
            # Use same mask_mode and ratios during validation for consistency
            _, preds, _ = model(
                val_images,
                mask_mode=current_mask_mode,
                mask_ratio=args.mask_ratio,  # used if mode='random'
                block_ratio=current_block_ratio,
                random_ratio=current_random_ratio,
                flipped=current_flipped,
                random_offset=current_random_offset
            )
        reconstructions = model.unpatchify(preds)

        # Compute batch-level metrics
        mse_loss = (reconstructions - val_images).pow(2).mean().item()
        ssim_score = ssim_metric(reconstructions, val_images).item()
        lpips_score = lpips_metric(reconstructions, val_images).mean().item()

        # Log metrics to W&B
        wandb.log({
            "val_MSE": mse_loss,
            "val_SSIM": ssim_score,
            "val_LPIPS": lpips_score,
            "epoch": epoch,
        })

        print(f"Epoch {epoch}: val_MSE={mse_loss:.4f}, val_SSIM={ssim_score:.4f}, val_LPIPS={lpips_score:.4f}")

        # Log a few examples to W&B as images
        num_log_images = min(val_images.size(0), 4)  # Log up to 4 images

        mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)

        for i in range(num_log_images):
            img = val_images[i].detach().cpu().numpy()
            recon = reconstructions[i].detach().cpu().numpy()

            # Denormalize for display
            img = np.clip(img * std + mean, 0, 1)
            recon = np.clip(recon * std + mean, 0, 1)

            img = np.transpose(img, (1, 2, 0))
            recon = np.transpose(recon, (1, 2, 0))

            # Vertical flip
            img = np.flipud(img)
            recon = np.flipud(recon)

            wandb.log({
                f"val/original_{i}": wandb.Image(img, caption=f"Val_Original_{i}"),
                f"val/reconstruction_{i}": wandb.Image(recon, caption=f"Val_Reconstruction_{i}"),
                "epoch": epoch
            })

        # Compute a simple validation metric (MSE) for logging
        val_loss = (reconstructions - val_images).pow(2).mean().item()
        wandb.log({
            "val_loss": val_loss,
            "epoch": epoch,
        })

    # Final Validation Phase (if args.eval)
    if args.eval:
        print("Running final evaluation on validation set...")
        model.eval()
        for val_images in data_loader_val:
            val_images = val_images.to(device, non_blocking=True)
            with torch.no_grad():
                _, preds, _ = model(val_images, mask_ratio=args.mask_ratio)
            reconstructions = model.unpatchify(preds)
            # Here you can do final evaluation steps if needed
        print("Final evaluation complete.")

    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    wandb.finish()

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)