import cv2
import os

# Source directory path containing input images
input_folder = 'F:\\DeepLearning\\dset-s2\\archive\\Water Bodies Dataset\\Masks_png'
# Output directory path for grayscale images
output_folder = 'F:\\DeepLearning\\dset-s2\\archive\\Water Bodies Dataset\\Masks_gray'

# Ensure output directory exists; create if not present
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process all files in input directory
for filename in os.listdir(input_folder):
    # Construct full input path
    input_path = os.path.join(input_folder, filename)

    # Check for common image file extensions (case-insensitive)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # Read image using OpenCV (BGR color format by default)
        image = cv2.imread(input_path)

        # Convert to grayscale using BGR2GRAY color space transformation
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Construct output path preserving original filename
        output_path = os.path.join(output_folder, filename)

        # Save grayscale image with original format (determined by extension)
        cv2.imwrite(output_path, gray_image)

        print(f"Converted and saved: {output_path}")

print("All images processed and converted to grayscale.")