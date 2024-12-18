# Facade-Autoencoder

This repository implements and extends Masked Autoencoders (MAE), an advanced self-supervised learning method for image reconstruction and representation learning, originally introduced in the paper "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2022) to reconstruct Venician facades.

We provide two implementations:

-Custom MAE: A fully customizable implementation of the MAE architecture, designed to explore various model components, such as masking strategies (block-based masking), patch sizes, embedding dimensions, and loss functions (MSE + Perceptual Loss).

-Pretrained MAE: Fine-tuning the official MAE model from Facebook AI Research on a specific dataset of architectural facades.

## References

This project builds upon the following repositories:

- [MAE by IcarusWizard](https://github.com/IcarusWizard/MAE): A custom implementation of Masked Autoencoders (MAE) for image reconstruction tasks.
- [Original MAE by Facebook AI Research](https://github.com/facebookresearch/mae): The official implementation of the Masked Autoencoder architecture described in the paper *"Masked Autoencoders Are Scalable Vision Learners"*.

