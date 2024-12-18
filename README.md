# Facade-Autoencoder

This repository implements and extends Masked Autoencoders (MAE), an advanced self-supervised learning method for image reconstruction and representation learning, originally introduced in the paper "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2022) to reconstruct Venician facades.

We explore the architectural and typological properties of Venetian facades through a series of advanced textural and color analyses. Specifically, we utilize Local Binary Patterns (LBP) to capture micro-level texture variations, Histogram of Oriented Gradients (HOG) to identify edge distributions and structural features, and Gabor filters to analyze texture frequency and orientation. These techniques provide a deeper understanding of the underlying patterns and visual characteristics of facades, offering valuable insights for tasks such as model selection, hyperparameter tuning, and error analysis in facade reconstruction and inpainting workflows. The extracted features help evaluate the suitability of different models, identify areas where models struggle to reconstruct intricate details, and optimize parameters for improved performance.

We provide two implementations:

- **Custom MAE:** A fully customizable implementation of the MAE architecture, designed to explore various model components, such as masking strategies (block-based masking), patch sizes, embedding dimensions, and loss functions (MSE + Perceptual Loss).

- **Pretrained and finetuned MAE:** Fine-tuning the official MAE model from Facebook AI Research on a specific dataset of architectural facades.

Refer to each the models individual READMEs for more details.

Additionally, we implement an NMF (Negative-Matrix Factorisation) model to analyze and decompose facade datasets into meaningful components. NMF factorizes the dataset, represented as a large matrix, into two smaller non-negative matrices: one representing the building blocks (components) of the data and the other describing the weights or contributions of these components to approximate the original data. The number of components is a tunable parameter, determined based on the desired level of detail and the specific application. Each NMF component typically captures localized patterns or features prevalent in facades, such as windows, edges, or balconies, which are then recombined to reconstruct the original structure.

## References

This project builds upon the following repositories:

- [MAE by IcarusWizard](https://github.com/IcarusWizard/MAE): A custom implementation of Masked Autoencoders (MAE) for image reconstruction tasks.
- [Original MAE by Facebook AI Research](https://github.com/facebookresearch/mae): The official implementation of the Masked Autoencoder architecture described in the paper *"Masked Autoencoders Are Scalable Vision Learners"*.

