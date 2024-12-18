import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from skimage.transform import resize
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import random_noise
from skimage.restoration import inpaint
from skimage.morphology import square
from skimage.filters import median
from skimage import exposure
import glob
import os

# ---------------------------
# 1. Data Preparation
# ---------------------------

def load_and_preprocess_images(image_paths, image_size=(256, 256)):
    images = []
    for img_path in image_paths:
        try:
            # Read image
            img = plt.imread(img_path)
            print(f"Loading image: {img_path}, original shape: {img.shape}")
            
            # Handle RGBA images
            if img.ndim == 3 and img.shape[2] == 4:
                print(f"Image has 4 channels (RGBA). Discarding the alpha channel.")
                img = img[..., :3]  # Keep only the first three channels (RGB)
            
            # Ensure the image has 3 channels
            if img.ndim == 3 and img.shape[2] == 3:
                # Convert to grayscale
                img = rgb2gray(img)
                print(f"Converted to grayscale, new shape: {img.shape}")
            elif img.ndim == 2:
                # Image is already grayscale
                print(f"Image is already grayscale.")
            else:
                print(f"Image has unexpected number of dimensions or channels: {img.ndim} dimensions, shape: {img.shape}")
                continue  # Skip this image
            
            # Resize image
            img_resized = resize(img, image_size, anti_aliasing=True)
            print(f"Resized image to {image_size}, new shape: {img_resized.shape}")
            
            # Normalize pixel values to [0, 1]
            if img_resized.max() > 0:
                img_normalized = img_resized / img_resized.max()
            else:
                img_normalized = img_resized
            images.append(img_normalized)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue  # Skip this image
    # Convert list to NumPy array
    images_array = np.array(images)
    print(f"Final images array shape: {images_array.shape}")
    return images_array

# Define the paths to your image directories
train_images_dir = 'data/completed_facades_full'  
test_images_dir = 'data/incomplete_facades/images'    

train_image_paths = glob.glob(os.path.join(train_images_dir, '*.png'))
print(f"Number of training images found: {len(train_image_paths)}")

# Load training images (complete facades)
train_images = load_and_preprocess_images(train_image_paths)

# Flatten images and create data matrix V
n_samples, img_height, img_width = train_images.shape
V = train_images.reshape(n_samples, -1)  # Shape: (num_pixels, num_samples)
print(f">>>>>>Training data shape (V): {V.shape}")  # Should be (n_samples, n_features)

# ---------------------------
# 2. Training the NMF Model
# ---------------------------

# Choose the number of components (features)
n_components = 400  # Adjust based on desired detail

# Initialize NMF model
# nmf_model = NMF(n_components=n_components, init='nndsvda', max_iter=500, random_state=42)
nmf_model = NMF(
    n_components=n_components,
    init='nndsvda',
    solver='mu',
    # solver='cd',
    tol=1e-4,
    max_iter=1000,
    verbose = 1 # Set verbosity level to 1
)


# Fit the model to the training data
W = nmf_model.fit_transform(V)  # W: (num_pixels, n_components)
H = nmf_model.components_       # H: (n_components, num_samples)

# ---------------------------
# 3. Reconstructing Incomplete Facades
# ---------------------------

# Load test images (incomplete facades)

test_image_paths = glob.glob(os.path.join(test_images_dir, '*.png'))


test_images = load_and_preprocess_images(test_image_paths[:20])
print(f"Number of test images found: {len(test_image_paths)}")

# Introduce missing data (if test images are not already incomplete)
def introduce_missing_data(images, missing_rate=0.2):
    incomplete_images = []
    masks = []
    for img in images:
        mask = np.random.choice([1, 0], size=img.shape, p=[1-missing_rate, missing_rate])
        incomplete_img = img * mask
        incomplete_images.append(incomplete_img)
        masks.append(mask)
    return np.array(incomplete_images), np.array(masks)

# Assuming test images are already incomplete; otherwise, uncomment below
# test_images, test_masks = introduce_missing_data(test_images)

# Flatten test images
test_images_flat = test_images.reshape(test_images.shape[0], -1)  # Shape: (num_samples, num_pixels)
print(f"Test data shape (test_images_flat): {test_images_flat.shape}") 

# Handle missing data
# Create masks where 1 indicates observed data and 0 indicates missing data
test_masks = np.where(test_images_flat > 0, 1, 0)  # Shape: (num_pixels, num_samples)

# Impute missing values (e.g., with zeros)
V_test_imputed = np.nan_to_num(test_images_flat)

# Reconstruct the images
H_test = nmf_model.transform(V_test_imputed)  # Shape: (num_samples, n_components)
V_test_reconstructed = np.dot(H_test, nmf_model.components_)  # Shape: (num_pixels, num_samples)

# Apply the mask to keep original observed pixels
# V_test_final = np.where(test_masks == 1, test_images_flat, V_test_reconstructed)

alpha = 0.8  # Weight for the original pixels
beta = 1.4   # Weight for the reconstructed pixels
V_test_final = alpha * test_images_flat + beta * (1 - test_masks) * V_test_reconstructed

# Reshape reconstructed images
reconstructed_images = V_test_final.reshape(-1, img_height, img_width)

# Adjust intensity after reconstruction to increase contrast
# reconstructed_images_intense = []

# for img in reconstructed_images:
#     # Apply contrast stretching
#     p2, p98 = np.percentile(img, (2, 98))
#     img_rescaled = exposure.rescale_intensity(img, in_range=(p2, p98))
#     reconstructed_images_intense.append(img_rescaled)

# reconstructed_images = np.array(reconstructed_images_intense)

# ---------------------------
# 4. Visualization
# ---------------------------

def display_images(original, incomplete, reconstructed, num_images=10, save_dir="saved_images"):
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(min(num_images, original.shape[0])):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # Original complete image (if available)
        axes[0].imshow(incomplete[i], cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        # Incomplete image
        axes[1].imshow(reconstructed[i], cmap='gray')
        axes[1].set_title('reconstructed Image')
        axes[1].axis('off')
        # # Reconstructed image
        # axes[2].imshow(reconstructed[i], cmap='gray')
        # axes[2].set_title('Reconstructed Image')
        # axes[2].axis('off')
        plt.show()
        filename = f'image_{i}.png'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid memory issues


# Assuming we have the original complete test images; if not, use test_images instead
original_test_images = test_images  # Replace with original images if available
incomplete_test_images = test_images  # Since test_images are incomplete
display_images(original_test_images, incomplete_test_images, reconstructed_images)

# ---------------------------
# 5. Evaluation (Optional)
# ---------------------------

from sklearn.metrics import mean_squared_error

def evaluate_reconstruction(original, reconstructed):
    mse = []
    for orig, recon in zip(original, reconstructed):
        mse.append(mean_squared_error(orig.flatten(), recon.flatten()))
    avg_mse = np.mean(mse)
    print(f'Average MSE over test images: {avg_mse}')

# Evaluate reconstruction
evaluate_reconstruction(original_test_images, reconstructed_images)