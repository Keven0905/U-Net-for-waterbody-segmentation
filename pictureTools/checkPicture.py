from PIL import Image
import os

def check_image(image_path):
    # Print the current image path being processed
    print(f"Checking image path: {image_path}")
    try:
        # Open and validate the image file
        with Image.open(image_path) as img:
            img.verify()  # Verify if the image file is valid and intact
            img.load()    # Load image data to ensure full readability
            width, height = img.size
            # Check for valid image dimensions
            if width <= 0 or height <= 0:
                print(f"Invalid image dimensions: {image_path}")
            else:
                print(f"Valid image: {image_path}")
    except Exception as e:
        # Handle exceptions occurred during image processing
        print(f"Error loading image: {image_path}, error: {e}")

# Iterate through all image files in the directory for validation
for filename in os.listdir("F:\\unet-pytorch-main\\VOCdevkit\\VOC2007\\JPEGImages"):
    # Filter JPEG files with case-insensitive extension check
    if filename.endswith('.jpg') or filename.endswith('.JPG'):
        file_path = os.path.join('F:\\unet-pytorch-main\\VOCdevkit\\VOC2007\\JPEGImages', filename)
        check_image(file_path)