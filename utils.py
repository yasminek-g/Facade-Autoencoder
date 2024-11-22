import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def read_feather_file(input_file):
    """
    Reads a Feather file and returns a GeoDataFrame.

    Parameters:
    - input_file (str): Path to the input Feather file.

    Returns:
    - gdf (GeoDataFrame): The loaded GeoDataFrame.

    Raises:
    - FileNotFoundError: If the input file does not exist.
    - ValueError: If the file cannot be read into a GeoDataFrame.
    """
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file {input_file} does not exist.")

    try:
        gdf = gpd.read_feather(input_file)
    except Exception as e:
        raise ValueError(f"Error reading {input_file}: {e}")

    return gdf


def save_geodataframe(gdf, output_file):
    """
    Saves the GeoDataFrame to a Feather file without the geometry column.

    Parameters:
    - gdf (GeoDataFrame): The GeoDataFrame to save.
    - output_file (str): Path to the output Feather file.
    """
    gdf_no_geometry = gdf.drop(columns=["geometry"], errors='ignore')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    gdf_no_geometry.to_feather(output_file)
    print(f"GeoDataFrame saved to {output_file} without geometry.")


def save_array(array, output_file):
    """
    Saves a NumPy array to a .npy file.

    Parameters:
    - array (ndarray): The array to save.
    - output_file (str): Path to the output .npy file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.save(output_file, array)
    # print(f"Array saved to {output_file}")


def load_images_from_directory(directory):
    """
    Loads .npy files from a directory and organizes them into a dictionary for visualization.

    Parameters:
    - directory (str): Path to the directory containing .npy files.

    Returns:
    - images_dict (dict): A dictionary where keys are filenames (without extension) and values are image arrays.
    """
    images_dict = {}
    for file in os.listdir(directory):
        if file.endswith(".npy"):
            file_path = os.path.join(directory, file)
            try:
                # Load the .npy file
                image = np.load(file_path)

                # Use the filename (without extension) as the dictionary key
                key = os.path.splitext(file)[0]
                images_dict[key] = image
            except Exception as e:
                print(f"Error loading {file}: {e}")

    return images_dict


def visualize_images(images_dict):
    num_images = len(images_dict)
    fig, axes = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

    if num_images == 1:
        axes = [axes]

    for ax, (title, image) in zip(axes, images_dict.items()):
        if image.ndim == 2:
            # Handle NaN values in depth maps
            cmap = plt.cm.viridis
            cmap.set_bad(color='gray')  # Set color for NaNs
            im = ax.imshow(image, cmap=cmap, origin='lower')
            plt.colorbar(im, ax=ax)
        else:
            ax.imshow(image.astype(np.uint8), origin='lower')
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
