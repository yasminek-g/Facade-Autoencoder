from tqdm import tqdm
from facade_extraction import *

def process_all_feathers(input_dir, full_output_dir, subfacades_output_dir):
    """
    Processes all Feather files in a directory using the `process_facade` function, 
    saving full facades and subfacades to separate directories.

    Parameters:
    - input_dir (str): Directory containing Feather files.
    - full_output_dir (str): Directory to save full facade .npy files.
    - subfacades_output_dir (str): Directory to save subfacade .npy files.
    """
    # Ensure output directories exist
    os.makedirs(full_output_dir, exist_ok=True)
    os.makedirs(subfacades_output_dir, exist_ok=True)

    # List all Feather files in the input directory
    feather_files = [file for file in os.listdir(input_dir) if file.endswith(".feather")]

    # Set up the progress bar
    with tqdm(total=len(feather_files), desc="Processing Feather files") as pbar:
        for file in feather_files:
            file_path = os.path.join(input_dir, file)
            base_name = os.path.splitext(file)[0]

            # Temporary directories for each process
            temp_full_dir = os.path.join(full_output_dir, "temp")
            temp_subfacades_dir = os.path.join(subfacades_output_dir, "temp")
            os.makedirs(temp_full_dir, exist_ok=True)
            os.makedirs(temp_subfacades_dir, exist_ok=True)

            try:
                # Use the `process_facade` function to process the facade
                process_facade(
                    input_file=file_path,
                    output_dir=temp_full_dir,
                    save_results=True,
                    visualize_results=False,
                    process_subfacades=True,
                    padded_subfacades=False
                )

                # Move full facades to the `full_output_dir`
                for temp_file in os.listdir(temp_full_dir):
                    if "full_rgb" in temp_file:
                        os.rename(
                            os.path.join(temp_full_dir, temp_file),
                            os.path.join(full_output_dir, temp_file)
                        )

                # Move subfacades to the `subfacades_output_dir`
                for temp_file in os.listdir(temp_full_dir):
                    if "subfacade" in temp_file:
                        os.rename(
                            os.path.join(temp_full_dir, temp_file),
                            os.path.join(subfacades_output_dir, temp_file)
                        )

            except Exception as e:
                print(f"Error processing {file}: {e}")

            finally:
                # Clean up temp directories
                for temp_dir in [temp_full_dir, temp_subfacades_dir]:
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)

            # Update progress bar
            pbar.update(1)


input_dir = "dataframes_feather"
full_output_dir = "full_npy"
subfacades_output_dir = "subfacades_npy"

process_all_feathers(input_dir, full_output_dir, subfacades_output_dir)
