import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    """DenseNet composite layer with bottleneck architecture
    Implements:
    - BatchNorm → ReLU → 1x1 Conv (bottleneck)
    - BatchNorm → ReLU → 3x3 Conv (feature production)
    - Optional dropout
    - Feature concatenation
    """

    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super().__init__()
        # Bottleneck layer reduces input channels before expansion
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False))

        # Feature production layer with growth rate expansion
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate

    def forward(self, x):
        """Implements feature concatenation across channels dimension
        Args:
            x: Input tensor of shape [N, C, H, W]
        Returns:
            Concatenated output tensor of shape [N, C+growth_rate, H, W]
        """
        new_features = super().forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)  # Channel-wise concatenation


class _DenseBlock(nn.Sequential):
    """Sequential container of multiple DenseLayers
    Each layer increases channel dimension by growth_rate
    """

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super().__init__()
        for i in range(num_layers):
            # Each layer receives concatenated features from all previous layers
            layer = _DenseLayer(
                num_input_features + i * growth_rate,  # Cumulative channels
                growth_rate,
                bn_size,
                drop_rate
            )
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    """Compression block between DenseBlocks
    Reduces channel count through:
    1. BatchNorm
    2. 1x1 convolution (channel compression)
    3. 2x2 average pooling (spatial downsampling)
    """

    def __init__(self, num_input_features, num_output_features):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """DenseNet architecture with customizable configurations
    Features:
    - Initial convolution + maxpool
    - Stacked DenseBlocks with Transition layers
    - Channel adjustment layers for feature pyramid networks
    - Pretrained weight loading compatibility
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, pretrained=False):
        super().__init__()

        # Initial feature extractor
        self.features = nn.Sequential(OrderedDict([
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True)),
            ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1))  # 1/4 spatial reduction
        ]))

        # Build DenseBlocks with Transition layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add DenseBlock
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features += num_layers * growth_rate  # Update channel count

            # Add Transition layer except after last block
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2  # 50% channel reduction
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Channel adjustment for feature pyramid networks
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=1),  # feat1: 64 → 64 (identity)
            nn.Conv2d(256, 128, kernel_size=1),  # feat2: 256 → 128
            nn.Conv2d(512, 256, kernel_size=1),  # feat3: 512 → 256
            nn.Conv2d(1024, 512, kernel_size=1),  # feat4: 1024 → 512
            nn.Conv2d(1024, 512, kernel_size=1)  # feat5: 1024 → 512
        ])

        # Load ImageNet pretrained weights
        if pretrained:
            arch = "densenet121" if block_config == (6, 12, 24, 16) else "densenet169"
            # Key renaming pattern for pretrained model compatibility
            pattern = re.compile(
                r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$")
            state_dict = model_zoo.load_url(f"https://download.pytorch.org/models/{arch}-afa5c3bd.pth")
            # State dict key adaptation for sequential modules
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            self.load_state_dict(state_dict, strict=False)  # Ignore final layers

        # Remove original classification head
        del self.features.norm5
        del self.features.relu5

    def forward(self, x):
        """Extracts multi-scale features for downstream tasks
        Returns:
            List of feature maps at different scales:
            - feat1: 1/4 resolution
            - feat2: 1/8 resolution
            - feat3: 1/16 resolution
            - feat4: 1/32 resolution
            - feat5: 1/32 resolution (after final block)
        """
        features = [x]  # Store intermediate features
        for name, module in self.features.named_children():
            x = module(x)
            # Capture outputs from critical layers
            if name in ["relu0", "denseblock1", "denseblock2", "denseblock3", "denseblock4"]:
                features.append(x)
                # Debug: print(f"Feature {name} shape: {x.shape}")

        # Channel adjustment for feature pyramid compatibility
        feat1 = self.channel_adjust[0](features[1])  # After initial conv block
        feat2 = self.channel_adjust[1](features[2])  # After first denseblock
        feat3 = self.channel_adjust[2](features[3])  # After second denseblock
        feat4 = self.channel_adjust[3](features[4])  # After third denseblock
        feat5 = self.channel_adjust[4](features[5])  # After fourth denseblock

        return [feat1, feat2, feat3, feat4, feat5]