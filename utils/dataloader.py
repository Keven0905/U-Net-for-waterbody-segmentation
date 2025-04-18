import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines = annotation_lines  # List of data samples
        self.length = len(annotation_lines)
        self.input_shape = input_shape  # Model input dimensions [H, W]
        self.num_classes = num_classes  # Number of semantic classes
        self.train = train  # Training mode flag
        self.dataset_path = dataset_path  # Root directory of VOC dataset

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]  # Extract filename without extension

        # -------------------------------#
        #   Data loading and preprocessing
        # -------------------------------#
        jpg = Image.open(os.path.join(self.dataset_path, "VOC2007/JPEGImages", name + ".jpg"))
        png = Image.open(os.path.join(self.dataset_path, "VOC2007/SegmentationClass", name + ".png"))

        # Apply data augmentation in training mode
        if self.train:
            jpg, png = self.get_random_data(jpg, png, self.input_shape, random=True)

        # Normalize image and convert to CxHxW format
        jpg = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2, 0, 1])
        png = np.array(png)

        # Handle out-of-range class indices (boundary pixels in VOC)
        png[png >= self.num_classes] = self.num_classes

        # -------------------------------#
        #   One-hot encoding with background class
        #   +1 accounts for boundary/void class
        # -------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((self.input_shape[0], self.input_shape[1], self.num_classes + 1))

        return jpg, png, seg_labels

    def rand(self, a=0, b=1):
        """Uniform random number generator for augmentation parameters"""
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        """Data augmentation pipeline including:
        - Random scaling and aspect ratio distortion
        - Horizontal flipping
        - Padding and position jitter
        - Color space transformations
        """
        # Ensure RGB format for color transformations
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))  # Maintain label as PIL Image

        # Extract dimensions
        iw, ih = image.size
        h, w = input_shape

        if not random:
            # Validation mode: simple center-cropping with aspect preservation
            scale = min(w / iw, h / ih)
            nw, nh = int(iw * scale), int(ih * scale)
            image = image.resize((nw, nh), Image.BICUBIC)
            label = label.resize((nw, nh), Image.NEAREST)

            # Center padding with gray values
            new_image = Image.new('RGB', [w, h], (128, 128, 128))
            new_label = Image.new('L', [w, h], (0))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # Random aspect ratio and scale augmentation
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)  # Bilinear for images
        label = label.resize((nw, nh), Image.NEAREST)  # Nearest neighbor for labels

        # Random horizontal flipping
        if self.rand() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # Position jitter with random offset
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image, label = new_image, new_label

        # Convert to OpenCV format for color transformations
        image_data = np.array(image, np.uint8)

        # HSV color space manipulation
        # Random coefficients for hue/saturation/value
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        h, s, v = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))

        # Create look-up tables for each channel
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)  # Hue is [0,179] in OpenCV
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        # Apply transformations and convert back to RGB
        image_data = cv2.merge((cv2.LUT(h, lut_hue),
                                cv2.LUT(s, lut_sat),
                                cv2.LUT(v, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        return image_data, label


def unet_dataset_collate(batch):
    """Custom collate function for DataLoader:
    - Stacks images, labels, and segmentation masks
    - Converts numpy arrays to PyTorch tensors
    - Maintains proper data types (float32 for images, long for class indices)
    """
    images, pngs, seg_labels = [], [], []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)

    # Convert to tensors with appropriate types
    images = torch.from_numpy(np.array(images)).float()  # FloatTensor [B,C,H,W]
    pngs = torch.from_numpy(np.array(pngs)).long()  # LongTensor [B,H,W]
    seg_labels = torch.from_numpy(np.array(seg_labels)).float()  # FloatTensor [B,H,W,C]

    return images, pngs, seg_labels