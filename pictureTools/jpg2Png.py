from PIL import Image
import os

# Source directory path containing original images
source_folder = 'F:\\DeepLearning\\dset-s2\\archive\\Water Bodies Dataset\\Masks'
# Destination directory path for converted PNG files
destination_folder = 'F:\\DeepLearning\\dset-s2\\archive\\Water Bodies Dataset\\Masks_png'

# Ensure destination directory exists; create if not present
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through all files in source directory
for filename in os.listdir(source_folder):
    # Check for JPG file extensions (case-insensitive)
    if filename.endswith('.jpg') or filename.endswith('.JPG'):
        # Construct full source path
        file_path = os.path.join(source_folder, filename)

        # Open source image using context manager
        with Image.open(file_path) as img:
            # Construct destination path with .png extension
            new_filename = os.path.splitext(filename)[0] + '.png'
            new_file_path = os.path.join(destination_folder, new_filename)

            # Save as PNG with lossless compression
            img.save(new_file_path, 'PNG')
            print(f'Converted {filename} to {new_filename}')

print('Conversion complete: All JPG files converted to PNG format.')