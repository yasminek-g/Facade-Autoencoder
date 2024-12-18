import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.filters import sobel
from sklearn.decomposition import PCA
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler


# Define the data directories
train_images_dir = 'data/complete_facades/images'
test_images_dir = 'data/incomplete_facades/images'

# Load image paths
train_image_paths = [os.path.join(train_images_dir, img) for img in os.listdir(train_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]
test_image_paths = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

# Image preprocessing function
def load_and_preprocess_images(image_paths, target_size=(512, 512), convert_gray=True):
    images = []
    for img_path in image_paths:
        img = imread(img_path)
        # Handle RGBA images
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert to grayscale
        if convert_gray and img.ndim == 3:
            img = rgb2gray(img)
        # Resize to target size
        img_resized = resize(img, target_size, anti_aliasing=True)
        images.append(img_resized)
    return np.array(images)

# Load and preprocess images
train_images = load_and_preprocess_images(train_image_paths)
test_images = load_and_preprocess_images(test_image_paths)

# Calculate basic statistics
def calculate_image_statistics(images):
    stats = {
        'mean': np.mean(images),
        'std': np.std(images),
        'min': np.min(images),
        'max': np.max(images)
    }
    return stats

train_stats = calculate_image_statistics(train_images)
test_stats = calculate_image_statistics(test_images)
print("Training Data Statistics:", train_stats)
print("Test Data Statistics:", test_stats)

# Edge detection to calculate edge density
def calculate_edge_density(images):
    edge_densities = []
    for img in images:
        edges = sobel(img)
        edge_density = np.sum(edges) / img.size
        edge_densities.append(edge_density)
    return np.mean(edge_densities), np.std(edge_densities)

train_edge_density = calculate_edge_density(train_images)
test_edge_density = calculate_edge_density(test_images)
print("Training Edge Density (Mean, Std):", train_edge_density)
print("Test Edge Density (Mean, Std):", test_edge_density)

# Apply PCA to determine intrinsic dimensionality
train_images_flat = train_images.reshape(train_images.shape[0], -1)
scaler = StandardScaler()
train_images_scaled = scaler.fit_transform(train_images_flat)

pca = PCA(n_components=100)
pca.fit(train_images_scaled)
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Plot explained variance
plt.figure(figsize=(10, 5))
plt.plot(explained_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.grid()
plt.show()

# Extract LBP features for texture analysis
def extract_lbp_features(images, radius=3, n_points=24):
    lbp_features = []
    for img in images:
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_features.append(lbp_hist / np.sum(lbp_hist))  # Normalize the histogram
    return np.array(lbp_features)

train_lbp_features = extract_lbp_features(train_images)
print("Sample LBP Features for Training Data:", train_lbp_features[:2])

# Determine NMF hyperparameters
n_components = np.argmax(explained_variance > 0.95) + 1  # Choose n_components to explain 95% variance
print(f"Suggested number of components for NMF: {n_components}")

