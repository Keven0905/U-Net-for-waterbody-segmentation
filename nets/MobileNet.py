import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone with channel adaptation layers for feature pyramid extraction
    Args:
        pretrained (bool): Load ImageNet pretrained weights
    """

    def __init__(self, pretrained=False):
        super().__init__()
        from torchvision.models import mobilenet_v2

        # Initialize official MobileNetV2 architecture
        self.base_model = mobilenet_v2(pretrained=pretrained).features

        # Channel adaptation layers for feature pyramid networks
        # Maps intermediate feature channels to target dimensions
        self.channel_adjust = nn.ModuleList([
            nn.Conv2d(16, 64, 1),  # Expansion layer 1: 16 → 64 channels
            nn.Conv2d(24, 128, 1),  # Inverted residual block 3 output
            nn.Conv2d(32, 256, 1),  # Intermediate feature adaptation
            nn.Conv2d(96, 512, 1),  # High-level feature transformation
            nn.Conv2d(320, 512, 1)  # Final feature map projection
        ])

    def forward(self, x):
        """Extracts multi-scale features with channel adaptation
        Returns:
            List[Tensor]: Feature maps at different scales:
            - feat1: [N, 64, 112, 112]
            - feat2: [N, 128, 56, 56]
            - feat3: [N, 256, 28, 28]
            - feat4: [N, 512, 14, 14]
            - feat5: [N, 512, 7, 7]
        """
        features = []
        # Critical layer indices for feature extraction:
        # [expand_conv, inverted_residual(3), inverted_residual(6), ...]
        layer_indices = [1, 3, 6, 13, 17]  # MobileNetV2 architecture checkpoints

        for i, layer in enumerate(self.base_model):
            x = layer(x)
            if i in layer_indices:
                features.append(x)

        # Channel dimension adaptation
        feat1 = self.channel_adjust[0](features[0])  # 16→64 @ 112x112
        feat2 = self.channel_adjust[1](features[1])  # 24→128 @ 56x56
        feat3 = self.channel_adjust[2](features[2])  # 32→256 @ 28x28
        feat4 = self.channel_adjust[3](features[3])  # 96→512 @ 14x14
        feat5 = self.channel_adjust[4](features[4])  # 320→512 @ 7x7

        return [feat1, feat2, feat3, feat4, feat5]