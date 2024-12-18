# Facade-Autoencoder

This repository implements and extends Masked Autoencoders (MAE), an advanced self-supervised learning method for image reconstruction and representation learning, originally introduced in the paper "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2022) to reconstruct Venician facades.

We provide two implementations:

-Custom MAE: A fully customizable implementation of the MAE architecture, designed to explore various model components, such as masking strategies (block-based masking), patch sizes, embedding dimensions, and loss functions (MSE + Perceptual Loss).

-Pretrained MAE: Fine-tuning the official MAE model from Facebook AI Research on a specific dataset of architectural facades.

## **Custom Masked Autoencoder**

This project implements a **Custom Masked Autoencoder (MAE)** from scratch for image reconstruction tasks. The model is tailored to reconstruct facade images with incomplete or missing regions, addressing specific challenges such as occlusion and data sparsity.

---

### **Overview**

This project explores training a custom **Masked Autoencoder (MAE)** for facade image reconstruction. Key highlights include:
- Training the MAE from scratch with **Masked MSE Loss** and **Perceptual Loss**.
- Data augmentation strategies for improved model generalization.
- Experiment tracking and visualization using **Weights & Biases (W&B)**.
- Handling small dataset limitations by introducing **fine-tuned pretrained models**.

The goal is to reconstruct high-quality images from masked inputs and analyze the impact of architectural and loss function choices.

---

### **Project Structure**

```plaintext
project/
│
├── data/                       # Directory for datasets
│   └── complete_npy/           # .npy images dataset
│
├── model_final.py              # Model implementation (Custom MAE)
├── utils.py                    # Utilities (loss functions, visualization)
├── train_mae.py                # Main training and validation script
│
├── requirements.txt            # Dependencies for the project
└── README.md                   # Project documentation
```


---

### **Features**

1. **Custom Masked Autoencoder (MAE)**:
   - Implemented with an encoder-decoder architecture.
   - Configurable masking strategy (50% masking ratio).

2. **Loss Functions**:
   - **Masked MSE Loss**: Focuses on masked patches.
   - **Perceptual Loss**: Improves perceptual quality using VGG19 features.

3. **Data Augmentation**:
   - Random horizontal flip, rotation, and color jittering for generalization.

4. **Experiment Tracking**:
   - Visualization of training progress, losses, and image reconstructions using **Weights & Biases (W&B)**.

---
## **Pre-trained MAE**

...

## **References**

This project builds upon the following repositories:

- [MAE by IcarusWizard](https://github.com/IcarusWizard/MAE): A custom implementation of Masked Autoencoders (MAE) for image reconstruction tasks.
- [Original MAE by Facebook AI Research](https://github.com/facebookresearch/mae): The official implementation of the Masked Autoencoder architecture described in the paper *"Masked Autoencoders Are Scalable Vision Learners"*.
















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

Run the following script to train the custom MAE:  

```bash
python train_MAE.py --batch_size 16 --total_epoch 1000 --output_model_path custom_mae.pth
```

### Visualizing Reconstruction Results

Reconstructed images and losses will be logged to **Weights & Biases (W&B)**. To enable W&B logging, ensure you have an account and log in using:  

```bash
wandb login
```

---

## **Masking Strategy**

This implementation uses **block-based masking**, which masks contiguous square patches of the input image. This better simulates real-world occlusion scenarios in facade datasets.

Example configuration in the model:  

```python
mask_ratio = 0.5  # 50% masking ratio
```

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

---

## **Future Directions**

- Implement a **region detection model** to identify missing facade parts.  
- Fine-tune the model on larger datasets with block masking pretraining.  
- Extend the model for real-world facade inpainting tasks.

---

## **References**

- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)  
- [Facebook MAE Implementation](https://github.com/facebookresearch/mae)

