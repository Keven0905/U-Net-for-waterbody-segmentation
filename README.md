# Keven0905
---
## Project Structure
```
U-Net-pytorch/
├── img/
├── logs/                           # Training log storage
├── miou_out/                       # Mean Intersection-over-Union outputs
├── model_data/                     # Pretrained model weights & configs
├── nets/                           # Network architectures
│   ├── densenet.py                 # DenseNet modules
│   ├── MobileNet.py                # MobileNet modules
│   ├── resnet.py                   # ResNet modules
│   ├── vgg.py                      # VGG modules
│   ├── unet.py                     # Core U-Net implementation
│   ├──unet_training.py             # Contains training utilities including loss functions 
├── pictureTools/                   # Image processing utilities
├── utils/                          # Framework utilities
│   ├── callbacks.py                # Training callbacks
│   ├── dataloader.py               # Data loading functions
│   ├── utils.py                    # Core utilities
│   ├── utils_fit.py                # Training routines
│   ├── utils_fit_bayes.py          # Bayesian optimization integration
│   └── utils_metrics.py            # Evaluation metrics
├── VOCdevkit/                      # Pascal VOC dataset
│   ├── ImageSets
│   ├── JPEGImages
│   └── SegmentationClass         
├── get_miou.py                     # mIoU calculation script
├── json_to_dataset.py              # Dataset conversion tool
├── predict.py                      # Inference script
├── requirements.txt                # Dependency list
├── summary.py                      # Model summary generator
├── train.py                        # Main training script
├── train_ACO.py                    # Ant Colony Optimization training
├── trainalpha.py                   # Alpha Optimization training
├── train_bayes_los.psy             # Bayesian loss optimization to search best alpha and beta
├── train_GA.py                     # Genetic Algorithm training
├── train_pso.py                    # Particle Swarm Optimization training
├── voc_annotation.py               # Dataset annotation tool
└── unet.py                         #
```

---
## Module Descriptions
### Nets Structure(`nets`)
- [`nets/densenet.py`](./densenet.py)
Implements the DenseNet architecture with dense blocks and transition layers for multi-scale feature extraction, supporting pretrained weight loading.
- [`nets/MobileNet.py`](./MobileNet.py)
Defines a MobileNetV2 backbone with channel adaptation layers to integrate features into a pyramid structure for downstream tasks.
- [`nets/resnet.py`](./resnet.py)
Provides ResNet variants ( ResNet50) using bottleneck blocks to extract hierarchical features for feature pyramid networks.
- [`nets/unet.py`](./unet.py)
Builds a U-Net model with CBAM attention and multi-backbone support (VGG, ResNet, etc.) for semantic segmentation tasks.
- [`nets/unet_training.py`](./unet_training.py)
Contains training utilities including loss functions (CE, Focal, Dice), weight initialization, and learning rate schedulers for U-Net optimization.
- [`nets/vgg.py`](./vgg.py)
Adapts VGG16 to return intermediate feature maps for feature pyramid applications, with pretrained weight support.
### Framework Utilities(`utils`)
- [`utils/callbacks.py`](./callbacks.py)
Implements training callbacks for loss tracking, metric evaluation (mIoU, F1-score), and visualization, along with model checkpointing and TensorBoard logging.
- [`utils/dataloader.py`](./dataloader.py)
Defines dataset handling, preprocessing, and data augmentation (scaling, flipping, HSV adjustments) for semantic segmentation tasks using VOC-format data.
- [`utils/utils.py`](./utils.py)
Provides utility functions for image processing (color conversion, resizing), reproducibility (seed setting), learning rate retrieval, and pretrained weight downloads.
- [`utils/utils_fit.py`](./utils_fit.py)
Manages complete training/validation cycles per epoch, including loss computation (CE/Focal + Dice), mixed-precision training, and model checkpointing.
- [`utils/utils_fit_bayes.py`](./utils_fit_bayes.py)
Extends training loops with configurable loss coefficients (λ, β) and multi-GPU support for flexible optimization strategies.
- [`utils/utils_metrics.py`](./utils_metrics.py)
Computes segmentation metrics (IoU, precision, recall, F-score), generates confusion matrices, and visualizes class-wise performance via plots.
### Training Function
- [`train.py`](train.py)
Main training script for U-Net segmentation with multi-backbone support, configurable training phases (freeze/unfreeze), and mixed-precision training.
  
   **Model Training Process**

  Before training, it is necessary to download the pre-trained weights first.
  
  Link: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A
  
  Extraction code: 6n2c
  
  After the download is completed, put the pre-training weight file into the model_data folder
  
  1.Using the Datasets I provided
  
  Put the voc dataset I provided into VOCdevkit and run voc_annotation.py.Run train.py for training.
  
  2.Using the Datasets your own
  
  Firsst,Note to use the VOC format for training.
  Before training, place the label file in the SegmentationClass under the VOC2007 folder in the VOCdevkit folder.
  place the image files in JPEGImages under the VOC2007 folder in the VOCdevkit folder.
  Generate the corresponding txt file using the voc_annotation.py file before training.
  Note to modify the num_classes of train.py to the number of categories plus 1.
  You can start training by running train.py.

- [`train_ACO.py`](train_ACO.py)
Implements Ant Colony Optimization (ACO) to automatically tune hyperparameters (learning rate, batch size, etc.) for segmentation model training.
- [`train_alpha.py`](train_alpha.py)
Uses an Alpha Evolution algorithm to optimize hyperparameters through adaptive mutation and path-guided exploration.
- [`train_bayes_loss.py`](train_bayes_loss.py)
Leverages Bayesian optimization (via Optuna) to balance CE/Focal and Dice loss weights for improved segmentation metrics (mIoU, F1).
- [`train_GA.py`](train_GA.py)
Applies Genetic Algorithm (GA) with tournament selection and mutation to evolve optimal hyperparameter combinations.
- [`train_pso.py`](train_pso.py)
Employs Particle Swarm Optimization (PSO) to search for best hyperparameters by simulating particle movement in a search space.
### Predict Function
- [`unet.py`](unet.py)
Implements a U-Net-based segmentation model with inference, visualization (mask blending), FPS calculation, ONNX export, and mIoU evaluation capabilities for semantic segmentation tasks.
- [`predict.py`](predict.py)
It serves as a multi-mode inference interface for a U-Net segmentation model, supporting image/video processing, batch prediction, ONNX conversion, and performance benchmarking across different operational scenarios.

   **Model Test(Predict) Process**
   First, replace the model_path in the unet.py file with the optimal.pth file obtained from training, and then run the predict.py file
## Environment Setup
We recommend using [Anaconda](https://www.anaconda.com/) to manage your Python environment. This project is based on **Python 3.11** with the following recommended configuration:

| Library         | Version     |
|----------------|-------------|
| Python          | 3.11        |
| CUDA            | 12.5        |
| cuDNN           |             |
| torch           |             |
| torchvision     |             |
| tensorboard     |             |
|scipy            | 1.2.1       |
|numpy            | 1.17.0      |
|matplotlib       | 3.1.2       |
|opencv_python    | 4.1.2.30    |
|tqdm             | 4.60.0      |
|Pillow           | 8.2.0       |
|h5py             | 2.10.0      |
|labelme          | 3.16.7      |

## Reference
https://github.com/ggyyzm/pytorch_segmentation  

https://github.com/bonlime/keras-deeplab-v3-plus

https://github.com/bubbliiiing/unet-pytorch
