# **Custom Masked Autoencoder (MAE)**

This repository contains a **Custom Masked Autoencoder (MAE)** implemented from scratch for image reconstruction tasks, specifically targeting facade datasets with occluded or incomplete regions. The model uses a Vision Transformer (ViT) architecture and incorporates **block masking strategies**, advanced loss functions, and data augmentation techniques.

---

## **Overview**

The **Masked Autoencoder (MAE)** is a self-supervised learning model that reconstructs missing regions in input images. This implementation was developed to address facade image reconstruction challenges by experimenting with:  

- **Block Masking**: Contiguous masking of patches to simulate occluded regions.  
- **Customizable Model Architecture**: Vision Transformer-based encoder-decoder with flexible depth and embedding dimensions.  
- **Advanced Loss Functions**: Masked MSE loss combined with perceptual loss (VGG19).  
- **Data Augmentation**: Improves generalization despite limited training data.  

The goal is to reconstruct high-quality facades by encouraging the model to focus on masked regions while preserving architectural features and textures.

---

## **Features**

- **Custom Encoder-Decoder Architecture**:  
    - Vision Transformer (ViT)-based encoder.  
    - Lightweight decoder for pixel-level reconstruction.  

- **Masking Strategies**:  
    - **Block Masking**: Masks contiguous square blocks of patches.  
    - Configurable masking ratios for experimentation.

- **Advanced Loss Functions**:  
    - **Masked MSE Loss**: Focuses on reconstructing only the masked patches.  
    - **Perceptual Loss**: Improves perceptual quality using high-level features from a pre-trained VGG19 network.  

- **Data Augmentation**:  
    - Random horizontal flips, rotations, and color jittering to increase robustness.  

- **Experiment Tracking**:  
    - Integration with **Weights & Biases (W&B)** for logging losses and visualizing image reconstructions.  

---

## **Project Structure**

```plaintext
project/
│
├── data/                       # Directory for datasets
│   └── complete_npy/           # Directory containing .npy images
│
├── model_final.py              # Custom MAE model implementation
├── utils.py                    # Utility functions (losses, visualizations)
├── train_MAE.py                # Main training and validation script
│
├── requirements.txt            # List of dependencies
└── README.md                   # Project documentation
```

---

## **Model Architecture**

The Custom MAE follows an **Encoder-Decoder** structure:  

### **Encoder**  
- **Patch Embedding**: Images are divided into non-overlapping patches.  
- **Transformer Layers**:  
    - **12 Transformer layers** with **4 attention heads**.  
    - Embedding dimension: **256**.  
- **Positional Embeddings**: Added to patch embeddings to retain spatial context.  

### **Decoder**  
- **Learnable Mask Tokens**: Masked patches are replaced with learnable tokens.  
- **Lightweight Decoder**:  
    - **6 Transformer layers** with **4 attention heads**.  
    - Embedding dimension: **256**.  
- **Prediction Head**: Projects latent representations back to pixel space for reconstruction.

---

## **Installation**

### Prerequisites

- Python 3.8 or higher  
- PyTorch >= 1.10  
- torchvision  
- numpy  
- matplotlib  
- timm  

### Install Dependencies

Run the following command to install all required packages:  

```bash
pip install -r requirements.txt
```

---

## **Usage**

### Training the Custom MAE

To initialize the model, choose you parameters in the folowing line in the train_MAE file:  

```bash
model = MAE_ViT(image_size=224, patch_size=16, emb_dim=256, encoder_layer=12,
                    encoder_head=4, decoder_layer=6, decoder_head=4, mask_ratio=0.5).to(device)
```

### Visualizing Reconstruction Results

Reconstructed images and losses will be logged to **Weights & Biases (W&B)**. To enable W&B logging, ensure you have an account and log in using:  

```bash
wandb login
```

---

## **Masking Strategy**

This implementation uses **block-based masking**, which masks contiguous square patches of the input image. This better simulates real-world occlusion scenarios in facade datasets.

---

## **Loss Functions**

The model combines two loss functions for reconstruction:

1. **Masked MSE Loss**:  
   Focuses on masked patches only, ignoring visible patches.  

2. **Perceptual Loss**:  
   Utilizes a pre-trained VGG19 network to compare high-level features of reconstructed and ground-truth images.  

The combined loss function is:  

\[
\mathcal{L} = 0.6 \cdot \text{MSE Loss} + 0.4 \cdot \text{Perceptual Loss}
\]

---

## **Results**

Results are visualized in **Weights & Biases** during training. Example outputs include:  
- Original vs. reconstructed images.  
- Training and validation loss curves.  

