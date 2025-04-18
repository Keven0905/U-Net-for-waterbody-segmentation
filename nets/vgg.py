import torch.nn as nn
from torch.hub import load_state_dict_from_url


class VGG(nn.Module):
    """VGG Architecture with Feature Pyramid Output
    Modified to return intermediate feature maps for FPN applications
    """

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        # Original classification head components
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        """Extracts features at different stages:
        Returns list of feature maps at scales:
        - feat1: 1/2 spatial resolution
        - feat2: 1/4
        - feat3: 1/8
        - feat4: 1/16
        - feat5: 1/32
        """
        feat1 = self.features[:4](x)  # First conv block
        feat2 = self.features[4:9](feat1)  # Second conv block
        feat3 = self.features[9:16](feat2)  # Third conv block
        feat4 = self.features[16:23](feat3)  # Fourth conv block
        feat5 = self.features[23:-1](feat4)  # Fifth conv block
        return [feat1, feat2, feat3, feat4, feat5]

    def _initialize_weights(self):
        """Kaiming initialization with ReLU activation for convolutions"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    """Constructs sequential layers from configuration template
    Args:
        cfg: Layer configuration (list of channel numbers and 'M' for maxpool)
        batch_norm: Enable batch normalization
    """
    layers = []
    for v in cfg:
        if v == 'M':
            # Spatial downsampling with stride 2
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # Convolutional block: 3x3 conv -> BN -> ReLU
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)] if batch_norm \
                else [conv2d, nn.ReLU(inplace=True)]
            in_channels = v  # Update channel dimension
    return nn.Sequential(*layers)


# Configuration D (VGG16):
# Spatial dimension progression (output sizes for 512x512 input):
# 512,512,64 -> 256,256,128 -> 128,128,256 -> 64,64,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels=3, **kwargs):
    """Constructs VGG16 model with pretrained weights
    Args:
        pretrained: Load ImageNet pretrained weights
        in_channels: Input channels (3 for RGB)
    Returns:
        Model with removed classification head for feature extraction
    """
    model = VGG(make_layers(cfgs["D"], batch_norm=False, in_channels=in_channels), **kwargs)

    if pretrained:
        # Load official pretrained weights
        state_dict = load_state_dict_from_url(
            "https://download.pytorch.org/models/vgg16-397923af.pth",
            model_dir="./model_data"
        )
        model.load_state_dict(state_dict)

    # Remove classification components for feature extraction
    del model.avgpool
    del model.classifier
    return model