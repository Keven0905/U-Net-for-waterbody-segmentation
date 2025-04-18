# --------------------------------------------------------#
#   This file is used for adjusting label formats
# --------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

# -----------------------------------------------------------------------------------#
#   Origin_SegmentationClass_path   Path to original segmentation labels
#   Out_SegmentationClass_path      Output path for processed labels
#                                   Processed labels will be grayscale images.
#                                   Note: Small output values may reduce visibility.
# -----------------------------------------------------------------------------------#
Origin_SegmentationClass_path = "F:\\unet-pytorch-main\\VOCdevkit\\VOC2007\\SegmentationClass"
Out_SegmentationClass_path = "F:\\DeepLearning\\simple卷积神经网络\\PyTorch_NN\\dataset\\mask"

# -----------------------------------------------------------------------------------#
#   Origin_Point_Value  Original pixel values in the labels
#   Out_Point_Value     Target pixel values for output
#                       Must maintain one-to-one correspondence between input and output values
#   Example 1:
#   Origin_Point_Value = np.array([0, 255]); Out_Point_Value = np.array([0, 1])
#   Maps original value 0 -> 0, 255 -> 1
#
#   Example 2 (multiple values):
#   Origin_Point_Value = np.array([0, 128, 255]); Out_Point_Value = np.array([0, 1, 2])
#
#   Example 3 (RGB array processing):
#   Origin_Point_Value = np.array([[0, 0, 0], [1, 1, 1]]); Out_Point_Value = np.array([0, 1])
#   Maps RGB [0,0,0] -> 0, [1,1,1] -> 1
# -----------------------------------------------------------------------------------#
Origin_Point_Value = np.array([0, 255])
Out_Point_Value = np.array([0, 1])

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)

    # ---------------------------#
    #   Process and remap labels
    # ---------------------------#
    png_names = os.listdir(Origin_SegmentationClass_path)
    print("Processing all segmentation labels:")
    for png_name in tqdm(png_names):
        # Load original label image
        png = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))
        w, h = png.size

        # Convert to array and initialize output
        png = np.array(png)
        out_png = np.zeros([h, w])

        # Perform pixel value remapping
        for i in range(len(Origin_Point_Value)):
            # Create boolean mask for current value
            mask = png[:, :] == Origin_Point_Value[i]
            # Handle RGB case: require all channels match
            if len(np.shape(mask)) > 2:
                mask = mask.all(-1)
            out_png[mask] = Out_Point_Value[i]

        # Save processed grayscale label
        out_png = Image.fromarray(np.array(out_png, np.uint8))
        out_png.save(os.path.join(Out_SegmentationClass_path, png_name))

    # -------------------------------------#
    #   Analyze pixel value distribution
    # -------------------------------------#
    print("\nAnalyzing pixel value distribution in output labels:")
    classes_nums = np.zeros([256], np.int)
    for png_name in tqdm(png_names):
        png_file_name = os.path.join(Out_SegmentationClass_path, png_name)
        if not os.path.exists(png_file_name):
            raise ValueError(f"Label image not found: {png_file_name} (Check path and file extension)")

        # Count pixel value occurrences
        png = np.array(Image.open(png_file_name), np.uint8)
        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    # Print statistical results
    print("\nPixel value distribution summary:")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Count"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)