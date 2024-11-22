from utils import *
from sklearn.linear_model import LinearRegression
from scipy.spatial import cKDTree, ConvexHull


def fill_missing_subfacade_labels(gdf):
    """
    Fills missing subfacade labels in the GeoDataFrame with 'None'.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame.

    Returns:
    - gdf (GeoDataFrame): The GeoDataFrame with missing subfacade labels filled.
    """
    gdf['subfacade'] = gdf['subfacade'].fillna('None')
    return gdf


def fill_nan_z_with_nearest(gdf, color_columns=['R', 'G', 'B']):
    """
    Fills NaN values in the 'z_pointcloud' column using the nearest valid z-coordinate.
    Only points with valid color data are eligible for depth filling.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame with 'z_pointcloud' and color columns.
    - color_columns (list): List of color channel columns to check for validity (default: ['R', 'G', 'B']).

    Returns:
    - gdf (GeoDataFrame): The updated GeoDataFrame with 'z_pointcloud' NaNs filled.
    """
    # Ensure required columns exist
    required_columns = ['z_pointcloud'] + color_columns
    if not all(col in gdf.columns for col in required_columns):
        raise ValueError(f"GeoDataFrame must contain the columns: {required_columns}")

    # Identify valid points for z-coordinates
    valid_z = gdf['z_pointcloud'].notna()
    valid_color = gdf[color_columns].notna().all(axis=1)
    valid_points = valid_z & valid_color

    # Points to fill: NaN z_pointcloud but valid color channels
    fill_points = ~gdf['z_pointcloud'].notna() & valid_color

    # If no points need filling, return early
    if not fill_points.any():
        print("No NaN z_pointcloud values to fill.")
        return gdf

    # Build KDTree using valid points
    valid_coords = gdf.loc[valid_points, ['x_bin', 'y_bin']].values
    valid_z_values = gdf.loc[valid_points, 'z_pointcloud'].values
    kdtree = cKDTree(valid_coords)

    # Query nearest neighbor for fill points
    fill_coords = gdf.loc[fill_points, ['x_bin', 'y_bin']].values
    distances, indices = kdtree.query(fill_coords)

    # Map nearest z values to fill points
    gdf.loc[fill_points, 'z_pointcloud'] = valid_z_values[indices]

    return gdf


def map_none_subfacade_points_to_nearest_subfacade(gdf):
    """
    Maps points with 'None' subfacade labels to the nearest subfacade using cKDTree.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame with filled subfacade labels.

    Returns:
    - gdf (GeoDataFrame): The updated GeoDataFrame with 'None' labels replaced.
    """
    # Separate points by subfacade
    subfacade_points = {}
    for subfacade in gdf['subfacade'].unique():
        if subfacade != 'None':
            subfacade_points[subfacade] = gdf[gdf['subfacade'] == subfacade][['x_bin', 'y_bin']].values

    # Points with 'None' subfacade
    none_points = gdf[gdf['subfacade'] == 'None'][['x_bin', 'y_bin']].values
    if len(none_points) == 0:
        print("No 'None' subfacade points to map.")
        return gdf

    # Prepare data for cKDTree
    subfacade_labels = []
    subfacade_coords = []
    for subfacade, coords in subfacade_points.items():
        subfacade_labels.extend([subfacade] * len(coords))
        subfacade_coords.extend(coords)

    subfacade_coords = np.array(subfacade_coords)
    subfacade_labels = np.array(subfacade_labels)

    # Build KDTree and query
    kdtree = cKDTree(subfacade_coords)
    distances, indices = kdtree.query(none_points)

    # Map 'None' points to nearest subfacade
    closest_subfacades = subfacade_labels[indices]
    gdf.loc[gdf['subfacade'] == 'None', 'subfacade'] = closest_subfacades

    return gdf


def extract_facade_image(gdf, value_columns=['red', 'green', 'blue'], padding=False, x_bin_to_idx=None, y_bin_to_idx=None):
    """
    Extracts a 2D binned image from the GeoDataFrame.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame.
    - value_columns (list): List of columns to use for pixel values.
    - padding (bool): Whether to pad the image to the size of the full facade.
    - x_bin_to_idx (dict, optional): Mapping from x_bin values to indices for padding.
    - y_bin_to_idx (dict, optional): Mapping from y_bin values to indices for padding.

    Returns:
    - image_array (ndarray): The extracted image array.
    - x_bin_to_idx (dict): Mapping from x_bin values to array indices.
    - y_bin_to_idx (dict): Mapping from y_bin values to array indices.
    """
    # Handle missing values
    gdf[value_columns] = gdf[value_columns].fillna(0)

    # Unique bins
    x_bin_unique = sorted(gdf['x_bin'].unique())
    y_bin_unique = sorted(gdf['y_bin'].unique())

    # Create mappings
    if padding and x_bin_to_idx is not None and y_bin_to_idx is not None:
        # Use provided mappings for padding
        x_bins = x_bin_to_idx.keys()
        y_bins = y_bin_to_idx.keys()
        width = len(x_bin_to_idx)
        height = len(y_bin_to_idx)
    else:
        x_bins = x_bin_unique
        y_bins = y_bin_unique
        x_bin_to_idx = {val: idx for idx, val in enumerate(x_bins)}
        y_bin_to_idx = {val: idx for idx, val in enumerate(y_bins)}
        width = len(x_bins)
        height = len(y_bins)

    # Initialize image array
    channels = len(value_columns)
    image_array = np.zeros((height, width, channels), dtype=np.float32)

    # Populate image array
    for _, row in gdf.iterrows():
        x_bin = row['x_bin']
        y_bin = row['y_bin']
        values = row[value_columns].values.astype(np.float32)

        x_idx = x_bin_to_idx.get(x_bin)
        y_idx = y_bin_to_idx.get(y_bin)

        if x_idx is not None and y_idx is not None:
            image_array[y_idx, x_idx] = values

    return image_array, x_bin_to_idx, y_bin_to_idx


def extract_depth_map(gdf, x_bin_to_idx, y_bin_to_idx):
    """
    Extracts a depth map from the point cloud data in the GeoDataFrame, fitting a plane
    at the outermost points of the facade.

    Parameters:
    - gdf (GeoDataFrame): The input GeoDataFrame.
    - x_bin_to_idx (dict): Mapping from x_bin values to array indices.
    - y_bin_to_idx (dict): Mapping from y_bin values to array indices.

    Returns:
    - depth_array (ndarray): The extracted depth map array.
    """
    # Ensure required columns
    required_columns = ['x_pointcloud', 'y_pointcloud', 'z_pointcloud']

    # Identify points with valid point cloud data
    idx_valid = gdf[required_columns].notnull().all(axis=1)

    # Proceed only if there are valid points
    if idx_valid.sum() == 0:
        raise ValueError("No valid point cloud data available for depth map extraction.")

    # Get valid points
    gdf_valid = gdf.loc[idx_valid]
    points = gdf_valid[['x_pointcloud', 'y_pointcloud', 'z_pointcloud']].values

    # Compute convex hull to find outermost points
    try:
        hull = ConvexHull(points[:, :2])  # Use x, y coordinates for 2D convex hull
        hull_points = points[hull.vertices]  # Extract points on the convex hull
    except Exception as e:
        raise ValueError(f"Error computing convex hull: {e}")

    # Fit a plane using the outermost points
    X = hull_points[:, :2]  # x and y coordinates
    z = hull_points[:, 2]  # z coordinates
    reg = LinearRegression().fit(X, z)
    a, b = reg.coef_
    c = -1
    d = reg.intercept_

    # Ensure proper orientation of the plane (e.g., z-axis positive)
    normal_vector = np.array([a, b, c])
    if normal_vector[2] < 0:  # If z component is negative, flip the normal
        a, b, c, d = -a, -b, -c, -d

    # Compute perpendicular depth for valid points
    gdf['perpendicular_depth'] = np.nan
    gdf.loc[idx_valid, 'perpendicular_depth'] = (
        a * gdf.loc[idx_valid, 'x_pointcloud'] +
        b * gdf.loc[idx_valid, 'y_pointcloud'] +
        c * gdf.loc[idx_valid, 'z_pointcloud'] + d
    ) / np.sqrt(a**2 + b**2 + c**2)

    # Initialize depth array with NaNs
    width = len(x_bin_to_idx)
    height = len(y_bin_to_idx)
    depth_array = np.full((height, width), np.nan, dtype=np.float32)

    # Populate depth array
    for _, row in gdf.iterrows():
        x_bin = row['x_bin']
        y_bin = row['y_bin']
        depth = row['perpendicular_depth']

        x_idx = x_bin_to_idx.get(x_bin)
        y_idx = y_bin_to_idx.get(y_bin)

        if x_idx is not None and y_idx is not None:
            depth_array[y_idx, x_idx] = depth

    return depth_array


def process_facade(
    input_file,
    output_dir="output",
    save_results=True,
    visualize_results=True,
    process_subfacades=True,
    padded_subfacades=True
):
    """
    Processes a facade from a Feather file to extract images.

    Parameters:
    - input_file (str): Path to the input Feather file.
    - output_dir (str): Directory to save the outputs.
    - save_results (bool): Whether to save the results to files.
    - visualize_results (bool): Whether to visualize the results.
    - process_subfacades (bool): Whether to process subfacades separately.
    - padded_subfacades (bool): Whether subfacade images are padded to full facade size.
    """
    # Read and preprocess data
    gdf = read_feather_file(input_file)
    gdf = fill_missing_subfacade_labels(gdf)

    # Step 1: Fill NaN z-pointcloud values with nearest valid values
    # gdf = fill_nan_z_with_nearest(gdf, color_columns=['red', 'green', 'blue'])

    # Step 2: Strip whitespace and determine subfacade configuration
    gdf.loc[:, 'subfacade'] = gdf['subfacade'].str.strip().str.replace(r'\s+', '', regex=True)
    unique_subfacades = gdf['subfacade'].unique()

    # Determine if there are real subfacades (beyond 'None' and '_1_')
    has_only_none_and_1 = set(unique_subfacades).issubset({'None', '_1_'})
    has_real_subfacades = not has_only_none_and_1

    # Step 3: Map 'None' points to nearest subfacade only if real subfacades exist
    if has_real_subfacades:
        gdf = map_none_subfacade_points_to_nearest_subfacade(gdf)

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Extract full facade image
    full_image, x_bin_to_idx, y_bin_to_idx = extract_facade_image(gdf)
    images_dict = {'Full Facade RGB': full_image}

    # Save full facade image
    if save_results:
        rgb_output_file = os.path.join(output_dir, f"{base_name}_full_rgb.npy")
        save_array(full_image, rgb_output_file)

    # Step 4: Process subfacades if enabled, there are subfacades, and they are meaningful
    if process_subfacades and has_real_subfacades:
        for subfacade in unique_subfacades:

            if subfacade == 'None':
                continue

            sub_gdf = gdf[gdf['subfacade'] == subfacade].copy()
            if len(sub_gdf) == 0:
                continue

            # Extract subfacade image
            if padded_subfacades:
                sub_image, _, _ = extract_facade_image(
                    sub_gdf, padding=True, x_bin_to_idx=x_bin_to_idx, y_bin_to_idx=y_bin_to_idx
                )
            else:
                sub_image, sub_x_bin_to_idx, sub_y_bin_to_idx = extract_facade_image(
                    sub_gdf, padding=False
                )

            images_dict[f"Subfacade {subfacade} RGB"] = sub_image

            # Save subfacade image
            if save_results:
                sub_rgb_output_file = os.path.join(
                    output_dir, f"{base_name}_subfacade_{subfacade}_rgb.npy"
                )
                save_array(sub_image, sub_rgb_output_file)

    # Step 5: Visualize results if enabled
    if visualize_results:
        visualize_images(images_dict)