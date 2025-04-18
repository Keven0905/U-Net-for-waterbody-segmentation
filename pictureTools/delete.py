import os

def delete_files_based_on_txt(txt_file_path, folder_path):
    """
    Delete files named 'water_body_<number>.png' in target folder based on serial numbers listed in a txt file
    :param txt_file_path: Path to the txt file containing serial numbers (one per line)
    :param folder_path: Path to the target folder containing PNG files
    """
    # Read serial numbers from txt file with UTF-8 encoding
    with open(txt_file_path, 'r') as file:
        numbers = file.readlines()

    # Normalize numbers by stripping whitespace and keeping as strings for filename matching
    numbers = [number.strip() for number in numbers]

    # Iterate through directory with early filtering for PNG files
    for filename in os.listdir(folder_path):
        # Check filename pattern using prefix and suffix constraints
        if filename.startswith("water_body_") and filename.endswith(".png"):
            # Extract numerical identifier between prefix and suffix
            # Format: "water_body_<NUMBER>.png" -> extract <NUMBER>
            file_number = filename[len("water_body_"):-len(".png")]

            # Validate against allow-list with string comparison
            if file_number in numbers:
                file_path = os.path.join(folder_path, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")  # Confirmation of deletion
            else:
                print(f"Skipped: {filename} (not in allow list)")
        else:
            # Log files that don't match expected naming convention
            print(f"Skipped: {filename} (invalid naming pattern)")

# Example usage with Windows path format
txt_file_path = 'delete.txt'  # Replace with actual txt file path
folder_path = 'F:\\unet-pytorch-main\\VOCdevkit\\VOC2007\\SegmentationClass'  # Replace with actual folder path

delete_files_based_on_txt(txt_file_path, folder_path)