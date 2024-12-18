#%%
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_missing_areas_with_filters(image_array, epsilon_factor, black_threshold, alpha_ratio, max_y_value):
    """
    Analyze missing areas (black regions) with area and y-coordinate filters.
    
    Parameters:
        image_array (numpy array): Input RGB image as a NumPy array.
        epsilon_factor (float): Proportionality constant for contour simplification.
                                Smaller values result in more detailed contours.
        black_threshold (int): Intensity threshold to classify black regions.
        alpha_ratio (float): Minimum area ratio (compared to total image area) to consider a region.
        max_y_value (int): Maximum y-coordinate value to consider a region. Ignore regions above this value.
    
    Returns:
        results (list): A list of dictionaries containing original and simplified contours,
                        their ratios, and bounding box positions for filtered missing regions.
        processed_image (numpy array): Image with contours drawn for visualization.
    """
    # Step 1: Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Step 2: Threshold to identify black regions
    _, binary_image = cv2.threshold(grayscale_image, black_threshold, 255, cv2.THRESH_BINARY_INV)

    # Step 3: Find all contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Image area
    image_area = image_array.shape[0] * image_array.shape[1]

    results = []
    processed_image = image_array.copy()

    for contour in contours:
        # Calculate contour area
        contour_area = cv2.contourArea(contour)

        # Calculate bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Skip regions with y-coordinate above max_y_value
        if max_y_value is not None and y < max_y_value: # TODO: Need to be more specifically modified.
            continue

        # Simplify contour using Ramer-Douglas-Peucker
        epsilon = epsilon_factor * cv2.arcLength(contour, closed=True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)

        # TODO: We can consider more contour properties to filter out unwanted regions here 
        
        
        # Compute the ratio of points
        original_length = len(contour)
        simplified_length = len(simplified_contour)
        ratio = simplified_length / original_length

        # Skip areas below alpha_ratio (more likely to be a shading of another facade)
        if ratio < alpha_ratio:
            continue

        # Store results
        results.append({
            "original_contour": contour,
            "simplified_contour": simplified_contour,
            "ratio": ratio,
            "bounding_box": (x, y, w, h),
            "area": contour_area
        })

        # Draw contours on the image for visualization
        cv2.drawContours(processed_image, [contour], -1, (0, 0, 255), 1)  # Original (red)
        cv2.drawContours(processed_image, [simplified_contour], -1, (255, 0, 0), 1)  # Simplified (blue)

    return results, processed_image


# Apply the function with area and y-coordinate filters
alpha_ratio = 0.3
max_y_value = 0  # Only consider regions with a y-coordinate below 80

image_array = "data/facades_npy_flipped/1_[uid_W001]__rgb.npy"
image_array = np.load(image_array)

filtered_results, filtered_image = analyze_missing_areas_with_filters(
    image_array, epsilon_factor=0.0001, black_threshold=10, alpha_ratio=alpha_ratio, max_y_value=max_y_value
)

# Visualize the filtered image
plt.imshow(filtered_image)
plt.title("Filtered Missing Areas (by RDP Ratio and Y-coordinate)")
plt.axis('off')
plt.show()

# Prepare results for display
filtered_results_summary = [
    {
        "Region": idx + 1,
        "Original Points": len(result['original_contour']),
        "Simplified Points": len(result['simplified_contour']),
        "Ratio": result['ratio'],
        "Bounding Box (x, y, w, h)": result['bounding_box'],
        "Area": result['area']
    }
    for idx, result in enumerate(filtered_results)
]

filtered_results_df = pd.DataFrame(filtered_results_summary)

print(filtered_results_df)
# %%
