import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew
import time

# Define the data directory
train_images_dir = 'data/complete_facades/images'

# Load image paths
train_image_paths = [
    os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))
]

# Image preprocessing function
def load_and_preprocess_images(image_paths, target_size=(256, 256)):
    images = []
    for img_path in image_paths:
        img = imread(img_path)
        # Handle RGBA images
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)
        # Resize to target size
        img_resized = resize(img, target_size, anti_aliasing=True)
        images.append(img_resized)
    return np.array(images)

# Load and preprocess images
train_images = load_and_preprocess_images(train_image_paths)

# Flatten images for NMF input
n_samples, img_height, img_width, n_channels = train_images.shape
train_images_flat = train_images.reshape(n_samples, -1)

# Extract HOG features for texture analysis
def extract_hog_features(images):
    hog_features = []
    for img in images:
        # Convert to grayscale for HOG
        gray_img = rgb2gray(img)
        hog_feature = hog(gray_img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
        hog_features.append(hog_feature)
    return np.array(hog_features)

# Extract HOG features from training images
train_hog_features = extract_hog_features(train_images)

# Extract Gabor features for texture analysis
def extract_gabor_features(images, frequencies=[0.1, 0.3, 0.5]):
    gabor_features = []
    for img in images:
        gray_img = rgb2gray(img)
        gabor_feature = []
        for frequency in frequencies:
            filt_real, _ = gabor(gray_img, frequency=frequency)
            gabor_feature.append(filt_real.mean())
        gabor_features.append(gabor_feature)
    return np.array(gabor_features)

# Extract Gabor features from training images
train_gabor_features = extract_gabor_features(train_images)

# Extract color histogram features
def extract_color_histogram_features(images, bins=32):
    color_hist_features = []
    for img in images:
        hist_r, _ = np.histogram(img[:, :, 0], bins=bins, range=(0, 1), density=True)
        hist_g, _ = np.histogram(img[:, :, 1], bins=bins, range=(0, 1), density=True)
        hist_b, _ = np.histogram(img[:, :, 2], bins=bins, range=(0, 1), density=True)
        color_hist_features.append(np.concatenate([hist_r, hist_g, hist_b]))
    return np.array(color_hist_features)

# Extract color histogram features from training images
train_color_hist_features = extract_color_histogram_features(train_images)

# Visualize HOG features in 2D space using PCA
pca_hog = PCA(n_components=2)
hog_features_2d = pca_hog.fit_transform(train_hog_features)
plt.figure(figsize=(10, 6))
plt.scatter(hog_features_2d[:, 0], hog_features_2d[:, 1], alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of HOG Features Using PCA')
plt.grid(True)
plt.savefig('hog_features_pca.png')
plt.show()

# Visualize Gabor features in 2D space using PCA
pca_gabor = PCA(n_components=2)
gabor_features_2d = pca_gabor.fit_transform(train_gabor_features)
plt.figure(figsize=(10, 6))
plt.scatter(gabor_features_2d[:, 0], gabor_features_2d[:, 1], alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Gabor Features Using PCA')
plt.grid(True)
plt.savefig('gabor_features_pca.png')
plt.show()

# Visualize Color Histogram features in 2D space using PCA
pca_color_hist = PCA(n_components=2)
color_hist_features_2d = pca_color_hist.fit_transform(train_color_hist_features)
plt.figure(figsize=(10, 6))
plt.scatter(color_hist_features_2d[:, 0], color_hist_features_2d[:, 1], alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of Color Histogram Features Using PCA')
plt.grid(True)
plt.savefig('color_hist_features_pca.png')
plt.show()

# # Compare different initializations and solvers for NMF
# inits = ['nndsvda', 'random']
# solvers = ['mu', 'cd']
# n_components = 30  # Number of components for NMF

# nmf_results = []

# for init in inits:
#     for solver in solvers:
#         print(f"Evaluating NMF with init='{init}' and solver='{solver}'")
#         start_time = time.time()
#         nmf_model = NMF(n_components=n_components, init=init, solver=solver, max_iter=500, random_state=42)
#         W = nmf_model.fit_transform(train_images_flat)
#         H = nmf_model.components_
#         end_time = time.time()
#         reconstruction = np.dot(W, H)
#         reconstruction_error = np.linalg.norm(train_images_flat - reconstruction, 'fro')
#         runtime = end_time - start_time
#         nmf_results.append({
#             'init': init,
#             'solver': solver,
#             'reconstruction_error': reconstruction_error,
#             'runtime': runtime
#         })
#         print(f"Reconstruction error: {reconstruction_error:.4f}, Runtime: {runtime:.2f} seconds\n")

# # Visualize NMF analysis results
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# # Plot reconstruction errors
# errors = [result['reconstruction_error'] for result in nmf_results]
# labels = [f"{result['init']}, {result['solver']}" for result in nmf_results]
# axes[0].bar(labels, errors, color='skyblue')
# axes[0].set_ylabel('Reconstruction Error')
# axes[0].set_title('NMF Reconstruction Error by Initialization and Solver')
# axes[0].tick_params(axis='x', rotation=45)

# # Plot runtimes
# runtimes = [result['runtime'] for result in nmf_results]
# axes[1].bar(labels, runtimes, color='lightcoral')
# axes[1].set_ylabel('Runtime (seconds)')
# axes[1].set_title('NMF Runtime by Initialization and Solver')
# axes[1].tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.savefig('nmf_analysis.png')
# plt.show()

# print("Extracted features and performed analysis.")
