import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
import pywt

def analyze_textures_and_materials(image_path, num_clusters=5):
    """
    Analyze texture and material properties of a facade image.

    Args:
        image_path (str): Path to the input facade image.
        num_clusters (int): Number of clusters for material grouping.

    Returns:
        dict: Dictionary containing texture descriptors, clustered materials, and LAB color image.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ### _______ Texture Analysis _______ ###
    ### Texture Feature Extraction
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Local Binary Patterns (LBP)
    # Captures local texture properties based on differences in pixel intensities.
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")

    # Gabor Filters
    # Extracts features sensitive to frequency and orientation, 
    # useful for detecting structural patterns like bricks and arches.
    gabor_responses = []
    frequencies = [0.1, 0.2, 0.3]
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        for frequency in frequencies:
            filt_real, filt_imag = gabor(gray_image, frequency=frequency, theta=theta)
            gabor_responses.append(filt_real)

    gabor_features = np.mean(np.stack(gabor_responses, axis=0), axis=0)

    # Discrete Wavelet Transform (DWT)
    # Decomposes the image into different frequency components 
    # to capture fine details.
    coeffs = pywt.dwt2(gray_image, 'haar')
    cA, (cH, cV, cD) = coeffs
    dwt_features = np.hstack((cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()))

    ### _______ Material Analysis _______ ###
    ### Color and Material Analysis
    # Convert image to CIELAB color space
    # better distinguish subtle material differences (e.g., stucco vs. stone).
    lab_image = rgb2lab(image_rgb)
    lab_reshaped = lab_image.reshape((-1, 3))

    # K-Means Clustering on Color
    # group similar colors into material clusters (e.g., terracotta tiles, white walls).
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    material_clusters = kmeans.fit_predict(lab_reshaped)
    clustered_image = material_clusters.reshape(image.shape[:2])

    ### _______ Texture Mapping (Aligning Textures with Semantic Regions) _______ ###
    # For demonstration, map the LBP and Gabor features to the clustered image
    texture_mapped = {
        "LBP": lbp,
        "Gabor": gabor_features,
        "Clustered_Materials": clustered_image,
        "LAB_Image": lab_image
    }

    return texture_mapped

# Example Usage
image_path = "data/complete_facades/images/35_[uid_W001]_.png"  # Replace with the path to your facade image
result = analyze_textures_and_materials(image_path)

# Visualize Results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Local Binary Patterns (LBP)")
plt.imshow(result["LBP"], cmap="gray")

plt.subplot(2, 2, 2)
plt.title("Gabor Features (Mean Response)")
plt.imshow(result["Gabor"], cmap="gray")

plt.subplot(2, 2, 3)
plt.title("Clustered Materials")
plt.imshow(result["Clustered_Materials"], cmap="viridis")

plt.subplot(2, 2, 4)
plt.title("LAB Color Space (L Channel)")
plt.imshow(result["LAB_Image"][:, :, 0], cmap="gray")

plt.tight_layout()
plt.show()