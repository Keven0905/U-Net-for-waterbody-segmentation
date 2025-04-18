import random
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------#
# Convert image to RGB format to prevent errors with grayscale
# Supports only RGB input, converts other formats to RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


# Aspect-preserving image resizing with padding
# Returns:
#   - Resized image with padding
#   - Original dimensions (nw, nh) before padding
# ---------------------------------------------------#
def resize_image(image, size):
    iw, ih = image.size  # Original dimensions
    w, h = size  # Target dimensions

    # Calculate scaling ratio
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)  # Scaled width
    nh = int(ih * scale)  # Scaled height

    image = image.resize((nw, nh), Image.BICUBIC)  # High-quality downsampling
    new_image = Image.new('RGB', size, (128, 128, 128))  # Gray padding
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))  # Center alignment

    return new_image, nw, nh


# ---------------------------------------------------#
# Retrieve current learning rate from optimizer
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# ---------------------------------------------------#
# Seed initialization for reproducibility
# Applies to:
#   - Python native random
#   - NumPy
#   - PyTorch (CPU/CUDA)
#   - cuDNN backend configurations
# ---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Deterministic algorithms
    torch.backends.cudnn.benchmark = False  # Disable auto-tuner


# ---------------------------------------------------#
# Worker initialization function for DataLoader
# Ensures unique seeds per worker process based on:
#   - worker_id: DataLoader worker index
#   - rank: Process rank in distributed training
#   - seed: Base random seed
# ---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed  # Unique seed per worker
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ---------------------------------------------------#
# Input normalization (0-1 range)
# ---------------------------------------------------#
def preprocess_input(image):
    image /= 255.0
    return image


# ---------------------------------------------------#
# Configuration display in tabular format
# ---------------------------------------------------#
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


# ---------------------------------------------------#
# Pretrained weights downloader
# Supported backbones:
#   - VGG16
#   - ResNet50
# Saves to model_data directory by default
# ---------------------------------------------------#
def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url

    # PyTorch official model URLs
    download_urls = {
        'vgg': 'https://download.pytorch.org/models/vgg16-397923af.pth',
        'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth'
    }
    url = download_urls[backbone]

    # Directory initialization
    os.makedirs(model_dir, exist_ok=True)

    # Download via PyTorch's hub utility
    load_state_dict_from_url(url, model_dir)