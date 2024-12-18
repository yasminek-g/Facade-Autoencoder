import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgba2rgb, rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.decomposition import NMF
import time
import pickle

# Define the data directories
train_images_dir = 'data/complete_facades/images'
test_images_dir = 'data/incomplete_facades/images'

# Load image paths
train_image_paths = [
    os.path.join(train_images_dir, img) 
    for img in os.listdir(train_images_dir) 
    if img.endswith(('.png', '.jpg', '.jpeg'))
]

# Image preprocessing function (grayscale + texture extraction handled later)
def load_and_preprocess_images(image_paths, target_size=(256, 256)):
    images = []
    for img_path in image_paths:
        img = imread(img_path)
        # Handle RGBA images
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)
        # Convert to grayscale
        gray_img = rgb2gray(img)
        # Resize to target size
        img_resized = resize(gray_img, target_size, anti_aliasing=True)
        images.append(img_resized)
    return np.array(images)

print("Step 1: Loading and preprocessing training images...")
train_images = load_and_preprocess_images(train_image_paths[:200])

# Load test image paths
test_image_paths = [
    os.path.join(test_images_dir, img) 
    for img in os.listdir(test_images_dir) 
    if img.endswith(('.png', '.jpg', '.jpeg'))
]

print("Step 2: Loading and preprocessing test images...")
test_images = load_and_preprocess_images(test_image_paths[:10])

print("Step 3: Generating LBP and Gabor masks for training images...")
expanded_images = []
for img in train_images:
    # Compute LBP mask
    lbp_mask = local_binary_pattern(img, P=24, R=3, method='uniform')
    lbp_mask = lbp_mask / lbp_mask.max()  # Normalize LBP mask to [0, 1]

    # Compute Gabor mask
    gabor_mask, _ = gabor(img, frequency=0.3)
    gabor_mask = (gabor_mask - gabor_mask.min()) / (gabor_mask.max() - gabor_mask.min())  # Normalize Gabor to [0, 1]

    # Stack grayscale image with LBP and Gabor masks: shape (h, w, 3)
    combined_img = np.dstack((img, lbp_mask, gabor_mask))
    expanded_images.append(combined_img)

expanded_images = np.array(expanded_images)  # shape: (n_samples, h, w, 3)
n_samples, img_height, img_width, n_channels = expanded_images.shape

# Reorder and flatten training data so channels are contiguous
# Transpose to (n_samples, channels, height, width)
expanded_images_reordered = np.transpose(expanded_images, (0, 3, 1, 2))
# Flatten each channel separately
gray_flat = expanded_images_reordered[:, 0, :, :].reshape(n_samples, img_height*img_width)
lbp_flat = expanded_images_reordered[:, 1, :, :].reshape(n_samples, img_height*img_width)
gabor_flat = expanded_images_reordered[:, 2, :, :].reshape(n_samples, img_height*img_width)
# Concatenate channels along columns: [Gray | LBP | Gabor]
expanded_images_flat = np.hstack((gray_flat, lbp_flat, gabor_flat))

print("Step 4: Generating LBP and Gabor masks for test images...")
test_expanded_images = []
for img in test_images:
    lbp_mask = local_binary_pattern(img, P=24, R=3, method='uniform')
    lbp_mask = lbp_mask / lbp_mask.max()  # Normalize

    gabor_mask, _ = gabor(img, frequency=0.3)
    gabor_mask = (gabor_mask - gabor_mask.min()) / (gabor_mask.max() - gabor_mask.min())

    combined_img = np.dstack((img, lbp_mask, gabor_mask))
    test_expanded_images.append(combined_img)

test_expanded_images = np.array(test_expanded_images)
test_samples = test_expanded_images.shape[0]

# Reorder and flatten test data the same way
test_expanded_images_reordered = np.transpose(test_expanded_images, (0, 3, 1, 2))
test_gray_flat = test_expanded_images_reordered[:, 0, :, :].reshape(test_samples, img_height*img_width)
test_lbp_flat = test_expanded_images_reordered[:, 1, :, :].reshape(test_samples, img_height*img_width)
test_gabor_flat = test_expanded_images_reordered[:, 2, :, :].reshape(test_samples, img_height*img_width)
test_expanded_images_flat = np.hstack((test_gray_flat, test_lbp_flat, test_gabor_flat))

# Modify the NMF model to incorporate semantic loss
class SemanticNMF(NMF):
    def __init__(self, n_components=30, init='nndsvda', solver='mu', max_iter=1000, random_state=None, alpha=0.9, beta=0.1):
        super().__init__(n_components=n_components, init=init, solver=solver, max_iter=max_iter, random_state=random_state)
        self.alpha = alpha
        self.beta = beta

    def fit_transform(self, X, y=None, **fit_params):
        W = super().fit_transform(X, y, **fit_params)
        H = self.components_
        self.semantic_loss(X, W, H)
        return W

    def semantic_loss(self, X, W, H):
        reconstruction = np.dot(W, H)
        
        # Separate grayscale and texture components
        pixel_count = img_height * img_width
        X_gray = X[:, :pixel_count]
        X_lbp_gabor = X[:, pixel_count:]
        reconstruction_gray = reconstruction[:, :pixel_count]
        reconstruction_lbp_gabor = reconstruction[:, pixel_count:]

        # Calculate Frobenius norm for grayscale and LBP/Gabor components
        gray_loss = np.linalg.norm(X_gray - reconstruction_gray, 'fro')
        texture_loss = np.linalg.norm(X_lbp_gabor - reconstruction_lbp_gabor, 'fro')

        # Weighted sum of grayscale and texture losses using self.alpha and self.beta
        total_loss = self.alpha * gray_loss + self.beta * texture_loss
        print(f"Semantic-aware Reconstruction Loss: {total_loss:.4f}")

# Use SemanticNMF with specified parameters
init = 'nndsvd'
solver = 'cd'
n_components = 200  # Number of components for NMF

model_filename = f"semantic_nmf_model_{init}_{solver}.pkl"
print("Step 5: Checking for existing model...")
if os.path.exists(model_filename):
    print(f"Step 5a: Loading pre-trained model from {model_filename} for init='{init}' and solver='{solver}'")
    with open(model_filename, 'rb') as model_file:
        nmf_model = pickle.load(model_file)
else:
    print(f"Step 5b: Training Semantic NMF model with init='{init}' and solver='{solver}'...")
    start_time = time.time()
    nmf_model = SemanticNMF(n_components=n_components, init=init, solver=solver, max_iter=1000, random_state=42, alpha=0.9, beta=0.1)
    W = nmf_model.fit_transform(expanded_images_flat)
    H = nmf_model.components_
    end_time = time.time()
    runtime = end_time - start_time
    with open(model_filename, 'wb') as model_file:
        pickle.dump(nmf_model, model_file)
    print(f"Step 5c: Model training complete. Model saved as {model_filename}")

print("Step 6: Calculating reconstruction for test images...")
W_test = nmf_model.transform(test_expanded_images_flat)
H = nmf_model.components_
reconstruction_test = np.dot(W_test, H)
# Reshape reconstructed test images back to (n_samples, h, w, 3)
# Remember we concatenated channels as gray, lbp, gabor in sequence
reconstructed_gray = reconstruction_test[:, :img_height*img_width].reshape(test_samples, img_height, img_width)
reconstructed_lbp = reconstruction_test[:, img_height*img_width:2*img_height*img_width].reshape(test_samples, img_height, img_width)
reconstructed_gabor = reconstruction_test[:, 2*img_height*img_width:].reshape(test_samples, img_height, img_width)
reconstructed_test_images = np.stack((reconstructed_gray, reconstructed_lbp, reconstructed_gabor), axis=-1)

print("Step 7: Visualizing the first 10 test images after reconstruction...")
# Visualize grayscale channel of reconstructed images alongside the original
fig, axes = plt.subplots(2, 10, figsize=(20, 4))
num_display = min(10, test_samples)
for i in range(num_display):
    # Original test image (just the grayscale channel)
    axes[0, i].imshow(test_images[i], cmap='gray')
    axes[0, i].set_title(f'Original {i+1}')
    axes[0, i].axis('off')
    
    # Reconstructed test image grayscale channel
    axes[1, i].imshow(reconstructed_test_images[i, :, :, 0], cmap='gray')
    axes[1, i].set_title(f'Reconstructed {i+1}')
    axes[1, i].axis('off')

plt.suptitle('Original and Reconstructed Test Images (First 10)')
plt.tight_layout()
plt.savefig('reconstructed_images.png')
plt.show()

print("Extraction, semantic factorization, and visualization complete.")