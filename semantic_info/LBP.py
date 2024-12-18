import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew

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

def compute_lbp_for_image(img, radius=3, n_points=24):
    """
    Compute LBP histogram and LBP image for a single image (can be multi-channel).
    """
    if img.ndim == 2:
        # Grayscale image
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        lbp_hist = lbp_hist / np.sum(lbp_hist)  # normalize
        return lbp_hist, lbp
    else:
        # Multi-channel image: compute LBP for each channel and combine
        channels = img.shape[2]
        combined_hist = []
        combined_lbp = np.zeros(img.shape[:2], dtype=float)
        for c in range(channels):
            lbp = local_binary_pattern(img[:, :, c], n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            combined_hist.extend(lbp_hist)
            combined_lbp += lbp
        # Average the LBP image over channels
        combined_lbp /= channels
        return np.array(combined_hist), combined_lbp

# Extract LBP features for a list of images
def extract_lbp_features(images, radius=4, n_points=30):
    lbp_features = []
    lbp_images = []
    for img in images:
        hist, lbp_img = compute_lbp_for_image(img, radius=radius, n_points=n_points)
        lbp_features.append(hist)
        lbp_images.append(lbp_img)
    return np.array(lbp_features), np.array(lbp_images)

# Extract LBP features and LBP images from training images
train_lbp_features, train_lbp_images = extract_lbp_features(train_images)

# Calculate statistics
mean_lbp = np.mean(train_lbp_features, axis=0)
std_lbp = np.std(train_lbp_features, axis=0)
kurtosis_lbp = kurtosis(train_lbp_features, axis=0)
skew_lbp = skew(train_lbp_features, axis=0)

# Visualize statistics
plt.figure(figsize=(14, 10))
plt.plot(mean_lbp, label='Mean', color='blue')
plt.plot(std_lbp, label='Standard Deviation', color='green')
plt.plot(kurtosis_lbp, label='Kurtosis', color='red')
plt.plot(skew_lbp, label='Skewness', color='purple')
plt.xlabel('LBP Pattern Bin')
plt.ylabel('Value')
plt.title('Statistics of LBP Features (Mean, Std, Kurtosis, Skewness)')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('lbp_statistics.png')
plt.show()

print("Mean of LBP features:", mean_lbp)
print("Standard deviation of LBP features:", std_lbp)
print("Kurtosis of LBP features:", kurtosis_lbp)
print("Skewness of LBP features:", skew_lbp)

# Determine whether to use L1 or L2 regularization
if np.mean(kurtosis_lbp) > 3 or np.mean(np.abs(skew_lbp)) > 1:
    print("Suggested regularization: L1 (sparse features)")
else:
    print("Suggested regularization: L2 (smooth features)")

# Use PCA to reduce dimensionality
pca = PCA(n_components=2)
lbp_features_2d = pca.fit_transform(train_lbp_features)

# Visualize LBP features in 2D space
plt.figure(figsize=(10, 6))
plt.scatter(lbp_features_2d[:, 0], lbp_features_2d[:, 1], alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of LBP Features Using PCA')
plt.grid(True)
plt.savefig('lbp_features_pca.png')
plt.show()

# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(lbp_features_2d)

# Visualize clusters
plt.figure(figsize=(10, 6))
for cluster_label in np.unique(clusters):
    cluster_points = lbp_features_2d[clusters == cluster_label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster_label + 1}', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D Visualization of LBP Features with KMeans Clustering')
plt.legend()
plt.grid(True)
plt.savefig('lbp_features_kmeans.png')
plt.show()

# Display all LBP histograms
plt.figure(figsize=(12, 8))
for lbp_hist in train_lbp_features:
    plt.plot(lbp_hist, alpha=0.3)
plt.xlabel('LBP Pattern Bin')
plt.ylabel('Frequency')
plt.title('LBP Histograms for All Training Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('lbp_histograms.png')
plt.show()

# Visualize average LBP histogram
average_lbp_hist = np.mean(train_lbp_features, axis=0)
plt.figure(figsize=(12, 6))
plt.bar(range(len(average_lbp_hist)), average_lbp_hist)
plt.xlabel('LBP Pattern Bin')
plt.ylabel('Average Frequency')
plt.title('Average LBP Histogram for All Training Images')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('average_lbp_histogram.png')
plt.show()

# Visualize LBP patterns on sample images
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, ax in enumerate(axes):
    if i < len(train_lbp_images):
        ax.imshow(train_lbp_images[i], cmap='gray')
        ax.set_title(f'LBP Visualization for Image {i + 1}')
        ax.axis('off')
plt.suptitle('LBP Pattern Visualization on Sample Images')
plt.savefig('lbp_pattern_visualization.png')
plt.show()

# Spatial mapping of LBP codes on a sample image
sample_image_index = 0
sample_image = train_images[sample_image_index]
sample_lbp_image = train_lbp_images[sample_image_index]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(sample_image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(sample_lbp_image, cmap='gray')
axes[1].set_title('Spatial Mapping of LBP Codes')
axes[1].axis('off')

plt.suptitle('Spatial Mapping of LBP Codes on Sample Image')
plt.savefig('spatial_mapping_lbp.png')
plt.show()

# If you want to try a different radius and n_points, just call the function again
train_lbp_features_large_radius, _ = extract_lbp_features(train_images, radius=5, n_points=32)

# Display histograms for larger radius/n_points
plt.figure(figsize=(12, 8))
for lbp_hist in train_lbp_features_large_radius:
    plt.plot(lbp_hist, alpha=0.3)

plt.xlabel('LBP Pattern Bin')
plt.ylabel('Frequency')
plt.title('LBP Histograms for All Training Images (Larger Radius and More Points)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('lbp_histograms_large_radius.png')
plt.show()

print("Extracted LBP features for training images.")