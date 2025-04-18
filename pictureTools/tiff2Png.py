from PIL import Image
import os

# Path to directory containing TIFF format input images
input_folder = 'F:\\DeepLearning\\dset-s2\\RGB\\train_truth'
# Output directory path for converted PNG images
output_folder = 'F:\\DeepLearning\\dset-s2\\RGB\\train_truth_png'

# Ensure output directory exists; create if necessary
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all files in input directory
for filename in os.listdir(input_folder):
    # Check for TIFF extensions (case-sensitive check)
    if filename.endswith('.tif') or filename.endswith('.tiff'):
        # Construct full source file path
        file_path = os.path.join(input_folder, filename)

        # Open image using context manager for resource safety
        with Image.open(file_path) as img:
            # Construct output path with .png extension replacement
            output_file_path = os.path.join(output_folder, filename[:-4] + '.png')
            # Convert to RGB color mode for PNG compatibility and save with lossless compression
            img.convert('RGB').save(output_file_path, 'PNG')

print('Conversion process completed successfully.')