
'''
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

import timm
from functools import partial

import wandb

from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_finetune import train_one_epoch
from util.datasets import build_dataset

# Ensure timm version compatibility
assert timm.__version__ == "0.3.2"

def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for reconstruction', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio for MAE')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--finetune', default='/Users/oscargoudet/Desktop/Facade-Autoencoder/chekpoint/mae_pretrain_vit_huge.pth', type=str,
                        help='Path to the pretrained checkpoint')
    parser.add_argument('--output_dir', default='./output_dir', help='Directory to save outputs')
    parser.add_argument('--log_dir', default='./output_dir', help='Directory to save logs')
    parser.add_argument('--data_path', default='/Users/oscargoudet/Desktop/complete_processed_npy', type=str,
                        help='dataset path')    
    parser.add_argument('--device', default='cuda', help='Device for training/testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--resume', default='', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    return parser

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


def main(args):
    # Initialize Weights & Biases
    wandb.init(project="mae-reconstruction", config=args)
    torch.cuda.set_per_process_memory_fraction(0.8, device=0)  # Limit GPU usage to 80% of available memory
    torch.cuda.set_per_process_memory_fraction(0.8, device=1)  # Limit GPU usage to 80% of available memory
    
    print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Arguments:\n{json.dumps(vars(args), indent=4)}")

    device = torch.device(args.device)
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
        checkpoint_path = '/Users/oscargoudet/Desktop/Facade-Autoencoder/chekpoint/mae_pretrain_vit_huge.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_scaler = NativeScaler()

    # Resume training if applicable
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from checkpoint: {args.resume}")

    # Logging
    log_writer = SummaryWriter(log_dir=args.log_dir) if args.log_dir else None

    # Training loop
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # Training Phase
        train_stats = train_one_epoch(
            model=model,
            criterion=None,  # Loss is computed internally in the model
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            args=args
        )

        # Save Model Checkpoint
        # After the training loop
        if args.output_dir:
            final_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs - 1,
            }, final_checkpoint_path)
            print(f"Final checkpoint saved at {final_checkpoint_path}")


        # Log Training Metrics
        wandb.log({
            "train_loss": train_stats["loss"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch": epoch,
        })

        # Validation Phase (Run after each epoch)
        print("Evaluating model on validation set...")
        model.eval()  # Set model to evaluation mode
        val_images, val_masks = next(iter(data_loader_val))
        val_images = val_images.to(device, non_blocking=True)
        val_masks = val_masks.to(device, non_blocking=True)

        with torch.no_grad():
            _, preds, _ = model(val_images, mask_ratio=args.mask_ratio, informative_mask=val_masks)
        reconstructions = model.unpatchify(preds)

        # Save visualizations
        print(f"Saving visualizations for epoch {epoch}...")
        save_visualizations(val_images, reconstructions, epoch, args.output_dir)

        # Compute Validation Metrics
        evaluate_stats = {
            "val_loss": (reconstructions - val_images).pow(2).mean().item()  # Example loss for validation
        }
        wandb.log({
            **evaluate_stats,
            "epoch": epoch,
        })

    # Final Validation Phase
    if args.eval:
        print("Running final evaluation on validation set...")
        model.eval()
        for val_images, val_masks in data_loader_val:
            val_images = val_images.to(device, non_blocking=True)
            val_masks = val_masks.to(device, non_blocking=True)

            with torch.no_grad():
                _, preds, _ = model(val_images, mask_ratio=args.mask_ratio, informative_mask=val_masks)
            reconstructions = model.unpatchify(preds)

            # Save final visualizations
            save_visualizations(val_images, reconstructions, "final", args.output_dir)

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
'''

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

import timm
from functools import partial

import wandb

from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_mae
from engine_finetune import train_one_epoch
from util.datasets import build_dataset

# Ensure timm version compatibility
assert timm.__version__ == "0.3.2"


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for reconstruction', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio for MAE')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--finetune',
                        default='/Users/oscargoudet/Desktop/Facade-Autoencoder/chekpoint/mae_pretrain_vit_huge.pth',
                        type=str, help='Path to the pretrained checkpoint')
    parser.add_argument('--output_dir', default='./output_dir', help='Directory to save outputs')
    parser.add_argument('--log_dir', default='./output_dir', help='Directory to save logs')
    parser.add_argument('--data_path', default='/Users/oscargoudet/Desktop/complete_processed_npy', type=str,
                        help='Dataset path')
    parser.add_argument('--device', default='mps', help='Device for training/testing')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader workers')
    parser.add_argument('--pin_mem', action='store_true', help='Pin memory in DataLoader')
    parser.add_argument('--resume', default='', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, help='Start epoch')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    return parser


def main(args):
    wandb.init(project="mae-reconstruction", config=args)

    print(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    print(f"Arguments:\n{json.dumps(vars(args), indent=4)}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cudnn.benchmark = True

    # Set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.backends.mps.manual_seed_all(seed)

    # Load datasets
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

    model = models_mae.mae_vit_large_patch16_dec512d8b(
        img_size=args.input_size,
        in_chans=3,
        norm_pix_loss=False
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location=device)
        print(f"Loading pretrained checkpoint from: {args.finetune}")
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained weights with message: {msg}")

        for param in model.patch_embed.parameters():
            param.requires_grad = False
        for blk in model.blocks:
            for param in blk.parameters():
                param.requires_grad = False

        print("Pretrained encoder layers frozen.")

    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {n_parameters / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from checkpoint: {args.resume}")

    log_writer = None

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model=model,
            criterion=None,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=scaler,
            args=args
        )
        if args.output_dir:
            final_checkpoint_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': args.epochs - 1,
            }, final_checkpoint_path)
            print(f"Final checkpoint saved at {final_checkpoint_path}")

        wandb.log({"train_loss": train_stats["loss"], "lr": optimizer.param_groups[0]["lr"], "epoch": epoch})

    total_time = time.time() - start_time
    print(f"Training completed in {str(datetime.timedelta(seconds=int(total_time)))}")
    wandb.finish()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
