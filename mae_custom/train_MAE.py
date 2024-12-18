import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
import math
from model_final import *
from utils import setup_seed

# Main script
if __name__ == '__main__':
    # Initialize W&B project
    wandb.init(project="mae-reconstruction", config={
        "learning_rate": 3e-4,
        "batch_size": 16,
        "mask_ratio": 0.54,
        "image_size": 224,
        "patch_size": 8,
        "total_epochs": 700
    })
    config = wandb.config

    # Setup arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--base_learning_rate', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--total_epoch', type=int, default=700)
    parser.add_argument('--warmup_epoch', type=int, default=150)
    parser.add_argument('--output_model_path', type=str, default='mae-vit-pretrained.pt')
    args = parser.parse_args()

    # Seed for reproducibility
    setup_seed(args.seed)


    # Path to dataset
    data_dir = '/home/goudet/myfiles/project/complete_npy_244'


    # Define transformations (add data augmentation for the train_transform)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomRotation(10),           # Random rotation within ±10 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jittering
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = NumpyImageDataset(data_dir, transform=None)  # Load all image paths


    # Split dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply specific transformations for train and validation datasets
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Initialize model
    model = MAE_ViT(image_size=224, patch_size=16, emb_dim=256, encoder_layer=12,
                    encoder_head=4, decoder_layer=6, decoder_head=4, mask_ratio=0.51).to(device)

    # Loss functions
    masked_mse_loss_fn = MaskedMSELoss()
    perceptual_loss_fn = PerceptualLoss(device=device)

    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)





# Training and validation loop
for epoch in range(args.total_epoch):
    # Training Phase
    model.train()
    train_losses = []
    train_images, train_reconstructed = None, None  # Placeholder for training images

    for img in tqdm(train_dataloader):
        img = img.to(device)
        predicted_img, mask = model(img)

        # Compute losses
        mse_loss = masked_mse_loss_fn(predicted_img, img, mask)
        perceptual_loss = perceptual_loss_fn(predicted_img, img, mask)
        loss = 0.6 * mse_loss + 0.4 * perceptual_loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        # Store the first batch for W&B logging
        if train_images is None:
            train_images = img.clone().detach()
            train_reconstructed = predicted_img.clone().detach()

    # Step the learning rate scheduler
    lr_scheduler.step()
    avg_train_loss = sum(train_losses) / len(train_losses)
    wandb.log({"Epoch": epoch,"MSE train Loss": mse_loss.item(),
               "Perceptual train Loss": perceptual_loss.item(), "Train Loss": avg_train_loss})

    # Validation Phase
    model.eval()
    val_losses = []
    val_images, val_reconstructed = None, None  # Placeholder for validation images

    with torch.no_grad():
        for img in tqdm(val_dataloader):
            img = img.to(device)
            predicted_img, mask = model(img)

            # Compute validation losses
            mse_loss = masked_mse_loss_fn(predicted_img, img, mask)
            perceptual_loss = perceptual_loss_fn(predicted_img, img, mask)/10
            loss = 0.6 * mse_loss + 0.4 * perceptual_loss

            val_losses.append(loss.item())

            # Store the first batch for W&B logging
            if val_images is None:
                val_images = img.clone().detach()
                val_reconstructed = predicted_img.clone().detach()

    avg_val_loss = sum(val_losses) / len(val_losses)
    wandb.log({"Epoch": epoch, "MSE Val Loss": mse_loss.item(),
               "Perceptual Val Loss": perceptual_loss.item(), "Validation Loss": avg_val_loss})

    # Log Images to W&B
    if epoch % 10 == 0:
        # Combine Training Images
        combined_train_images = [
            wandb.Image(combine_images_side_by_side(train_images[i], train_reconstructed[i]),
                        caption=f"Train Epoch {epoch}")
            for i in range(min(5, train_images.size(0)))
        ]

        # Combine Validation Images
        combined_val_images = [
            wandb.Image(combine_images_side_by_side(val_images[i], val_reconstructed[i]),
                        caption=f"Val Epoch {epoch}")
            for i in range(min(20, val_images.size(0)))
        ]

        # Log Combined Images to W&B
        wandb.log({
            "Training: Original vs Reconstructed": combined_train_images,
            "Validation: Original vs Reconstructed": combined_val_images
        })

    # Save model checkpoint
    if epoch % 10 == 0:
        torch.save(model.state_dict(), args.output_model_path)

wandb.finish()


