# Masked Autoencoder with Vision Transformer (MAE-ViT)

This repository contains an implementation of a Masked Autoencoder (MAE) with a Vision Transformer (ViT) backbone, adapted for various masking strategies and reconstruction tasks. The code is inspired by [facebookresearch](https://github.com/facebookresearch/mae) from Meta Platforms, Inc. and integrates components from the DeiT and timm libraries.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Masking Strategies](#masking-strategies)
7. [Model Configuration](#model-configuration)
8. [Functions and Methods](#functions-and-methods)
9. [License](#license)
10. [References](#references)

---

## Overview

**Masked Autoencoders (MAEs)** are self-supervised learning models that learn to reconstruct masked portions of input images. This implementation utilizes a **Vision Transformer (ViT)** as the backbone encoder and provides flexibility with multiple masking strategies, including random masking and block masking.

The primary goal is to train the encoder with a self-supervised reconstruction task, which can later be fine-tuned for downstream tasks. We choose to finetune it for the reconstruction of facades, using our dataset. 

---

## Features

- **Vision Transformer Backbone**: Utilizes a ViT with customizable depth, embedding dimensions, and attention heads.
- **Flexible Masking Strategies**: Supports:
  - Random masking
  - Block masking (contiguous patches)
  - Combined masking (block + random)
- **Pretrained Encoder Blocks**: Ability to freeze pretrained encoder weights and train additional blocks.
- **Decoder for Reconstruction**: A lightweight decoder that reconstructs masked patches back to images.
- **Weight Initialization**: Positional embeddings and model weights are initialized with best practices (e.g., sinusoidal embeddings, Xavier initialization).
- **Dynamic Masking**: Configurable masking ratios and strategies during training.

---

## Architecture

### Encoder

- **Patch Embedding**: Converts images into a sequence of flattened patches.
- **Transformer Blocks**: Pretrained transformer blocks that can be frozen during training.
- **Additional Blocks**: New trainable transformer blocks for fine-tuning.

### Decoder

- **Linear Projection**: Embeds the latent tokens into the decoder space.
- **Transformer Blocks**: A series of transformer blocks to process the latent tokens.
- **Prediction Head**: Reconstructs the masked patches back to image space.

### Masking Strategies

- **Random Masking**: Randomly masks a specified percentage of patches.
- **Block Masking**: Masks contiguous vertical or square blocks of patches.
- **Combined Masking**: Uses a mix of block and random masking for more complex masking patterns.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- `timm` library (version 0.3.2)

### Install Dependencies

```bash
pip install torch torchvision timm numpy matplotlib
pip install torchmetrics lpips
```

---
## Usage

### Import the Model

```python
from models_mae import MaskedAutoencoderViT, mae_vit_large_patch16

# Initialize the model
model = mae_vit_large_patch16(img_size=224, patch_size=16)
```

### Forward pass
```python
# Input tensor of shape [batch_size, 3, 224, 224]
imgs = torch.randn(4, 3, 224, 224)

# Forward pass with random masking
loss, pred, mask = model(imgs, mask_mode='random', mask_ratio=0.75)

print(f"Reconstruction Loss: {loss.item()}")
```
---
## Masking strategies
## 1. Random masking
- Masks patches randomly.
```python
loss, pred, mask = model(imgs, mask_mode='random', mask_ratio=0.75)
```

## 2. Block masking
- Masks contiguous vertical blocks of patches.
### Block Masking Parameters

- **`block_ratio`**:  
  Fraction of rows (or patches) to mask.  
  - *Type*: `float`  
  - *Example*: `block_ratio=0.5` masks 50% of the rows.

- **`flipped`**:  
  If `True`, the row 0 is considered the bottom of the image; otherwise, row 0 is the top.  
  - *Type*: `bool`  
  - *Example*: `flipped=True` masks from the bottom up.

- **`random_offset`**:  
  Randomizes the starting position of the block vertically.  
  - *Type*: `bool`  
  - *Example*: `random_offset=True` selects a random vertical position for the block.

**Usage Example**:

```python
loss, pred, mask = model(imgs, mask_mode='block', block_ratio=0.5, flipped=True, random_offset=False)
```

## 3. Combined masking
- Combines block and random masking.
### Combined Masking Parameters

- **`block_ratio`**:  
  Fraction of patches to mask using block masking.  
  - *Type*: `float`  
  - *Example*: `block_ratio=0.3` masks 30% of the patches in a contiguous block.

- **`random_ratio`**:  
  Fraction of patches to mask using random masking from the remaining unmasked patches after block masking.  
  - *Type*: `float`  
  - *Example*: `random_ratio=0.45` masks 45% of the remaining patches randomly.

- **`flipped`**:  
  If `True`, the row 0 is considered the bottom of the image; otherwise, row 0 is the top.  
  - *Type*: `bool`  
  - *Example*: `flipped=True` masks from the bottom up.

- **`horizontal_random_offset`**:  
  Randomizes the horizontal position of the block during block masking.  
  - *Type*: `bool`  
  - *Example*: `horizontal_random_offset=True` selects a random horizontal position for the block.

- **`random_offset`**:  
  Randomizes the starting position of the block vertically during block masking.  
  - *Type*: `bool`  
  - *Example*: `random_offset=True` selects a random vertical position for the block.

**Usage Example**:

```python
loss, pred, mask = model(
    imgs, 
    mask_mode='combined', 
    block_ratio=0.3, 
    random_ratio=0.45, 
    flipped=True, 
    horizontal_random_offset=True, 
    random_offset=False
)
```

---
## Model configuration

The **large model** configuration for the Masked Autoencoder with Vision Transformer (MAE-ViT) has the following specifications:

- **Patch Size**: `16x16`  
- **Embedding Dimension**: `1024`  
- **Encoder Depth**: `24` layers  
- **Number of Attention Heads**: `16`  
- **Decoder Embedding Dimension**: `512`  
- **Decoder Depth**: `12` layers  
- **Decoder Attention Heads**: `16`  
- **MLP Ratio**: `4`  
- **Normalization Layer**: `LayerNorm` with `eps=1e-6`  

### Example Initialization

To initialize the large model with an image size of `224x224` and a patch size of `16x16`:

```python
from models_mae import mae_vit_large_patch16

model = mae_vit_large_patch16(img_size=224, patch_size=16)
```
---
## Functions and Methods

### `MaskedAutoencoderViT` Class

This class implements a Masked Autoencoder with a Vision Transformer backbone.

---

### Initialization

- **`initialize_weights()`**  
  Initializes the model weights and positional embeddings.

- **`freeze_pretrained_weights()`**  
  Freezes the weights of the pretrained encoder blocks to prevent updates during training.

---

### Data Processing

- **`patchify()`**  
  Converts input images into a sequence of flattened patches.

- **`unpatchify()`**  
  Reconstructs images from a sequence of patches.

---

### Masking Functions

- **`random_masking()`**  
  Applies random masking to the input patches.

- **`block_masking()`**  
  Applies block masking by masking contiguous vertical blocks of patches.

- **`combined_masking()`**  
  Combines block masking and random masking strategies.

---

### Forward Methods

- **`forward_encoder_dynamic()`**  
  Encoder forward pass with dynamic masking strategies.

- **`forward_decoder()`**  
  Decoder forward pass to reconstruct masked patches.

- **`forward_loss()`**  
  Computes the reconstruction loss between the original and predicted patches.

- **`forward()`**  
  Full forward pass, including encoding, decoding, and loss computation.
