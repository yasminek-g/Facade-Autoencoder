# Facade-Autoencoder
# **Custom Masked Autoencoder for Image Reconstruction**

This project implements a **Custom Masked Autoencoder (MAE)** from scratch for image reconstruction tasks. The model is tailored to reconstruct facade images with incomplete or missing regions, addressing specific challenges such as occlusion and data sparsity.

---

## **Table of Contents**
1. [Overview](#overview)  
2. [Project Structure](#project-structure)  
3. [Features](#features)  
4. [Acknowledgements](#acknowledgements)

---

## **Overview**

This project explores training a custom **Masked Autoencoder (MAE)** for facade image reconstruction. Key highlights include:
- Training the MAE from scratch with **Masked MSE Loss** and **Perceptual Loss**.
- Data augmentation strategies for improved model generalization.
- Experiment tracking and visualization using **Weights & Biases (W&B)**.
- Handling small dataset limitations by introducing **fine-tuned pretrained models**.

The goal is to reconstruct high-quality images from masked inputs and analyze the impact of architectural and loss function choices.

---

## **Project Structure**

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

## **Features**

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

## **Acknowledgements**
This project is inspired by:

-The original Masked Autoencoder (MAE) paper: "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2021).
-PyTorch and TorchVision frameworks for deep learning.

For further details, refer to the official paper:
Masked Autoencoders Are Scalable Vision Learners
