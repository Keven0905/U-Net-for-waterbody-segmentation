import torch
import torch.nn as nn
from nets.resnet import resnet50
from nets.vgg import VGG16
from nets.densenet import DenseNet
from nets.MobileNet import MobileNetV2


class CBAM(nn.Module):
    """Convolutional Block Attention Module with Dynamic Gating
    Components:
    - Channel attention with adaptive feature fusion
    - Spatial attention with dynamic gate control
    """

    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel attention component
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP for channel attention
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )

        # Dynamic channel gating mechanism
        self.channel_gate = nn.Sequential(
            nn.Linear(2 * channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        self.channel_sigmoid = nn.Sigmoid()

        # Spatial attention component
        self.spatial_gate = nn.Conv2d(2, 2, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid_gate = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # ===== Channel Attention =====
        # Extract pooled features
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))

        # Dynamic feature fusion
        gate_input = torch.cat([self.avg_pool(x).view(b, c),
                                self.max_pool(x).view(b, c)], dim=1)
        fusion_gate = self.channel_gate(gate_input)
        channel_att = fusion_gate * avg_out + (1 - fusion_gate) * max_out
        channel_att = self.channel_sigmoid(channel_att).view(b, c, 1, 1)

        # Apply channel-wise refinement
        x = x * channel_att.expand_as(x)

        # ===== Spatial Attention =====
        # Aggregate spatial features
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_feat = torch.cat([avg_out, max_out], dim=1)

        # Spatial gating mechanism
        spatial_gate = self.spatial_sigmoid_gate(self.spatial_gate(spatial_feat))
        gated_feat = spatial_gate * spatial_feat

        # Generate spatial attention weights
        spatial_att = self.spatial_sigmoid(self.spatial_conv(gated_feat))

        # Apply spatial refinement
        x = x * spatial_att

        return x


class unetUp(nn.Module):
    """U-Net Decoder Block with Attention
    Features:
    - Bilinear upsampling
    - Feature concatenation
    - CBAM-enhanced feature refinement
    """

    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_size)  # Integrated attention module

    def forward(self, inputs1, inputs2):
        # Feature fusion and processing
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.cbam(outputs)  # Attention-based refinement
        outputs = self.relu(outputs)
        return outputs


class Unet(nn.Module):
    """U-Net Architecture with Multi-backbone Support
    Supported backbones: VGG16, ResNet50, DenseNet, MobileNetV2
    Features:
    - Feature pyramid integration
    - Optional post-processing convolution
    - Backbone freezing capability
    """

    def __init__(self, num_classes=21, pretrained=False, backbone='vgg'):
        super(Unet, self).__init__()
        # Backbone initialization
        if backbone == 'vgg':
            self.vgg = VGG16(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "resnet50":
            self.resnet = resnet50(pretrained=pretrained)
            in_filters = [192, 512, 1024, 3072]
        elif backbone == "densenet":
            self.densenet = DenseNet(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        elif backbone == "mobilenet":
            self.mobilenet = MobileNetV2(pretrained=pretrained)
            in_filters = [192, 384, 768, 1024]
        else:
            raise ValueError(f'Unsupported backbone: {backbone}. Use vgg, resnet50, densenet, mobilenet')

        # Decoder configuration
        out_filters = [64, 128, 256, 512]
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # ResNet-specific post-processing
        if backbone == 'resnet50':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        # Final classification layer
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)
        self.backbone = backbone

    def forward(self, inputs):
        # Feature extraction
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet50":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)
        elif self.backbone == "densenet":
            [feat1, feat2, feat3, feat4, feat5] = self.densenet.forward(inputs)
        elif self.backbone == "mobilenet":
            [feat1, feat2, feat3, feat4, feat5] = self.mobilenet.forward(inputs)

        # Decoder processing
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        # Post-processing for ResNet
        if self.up_conv is not None:
            up1 = self.up_conv(up1)

        # Final prediction
        return self.final(up1)

    def freeze_backbone(self):
        """Immobilize backbone parameters for transfer learning"""
        backbone = getattr(self, self.backbone)
        for param in backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Activate backbone parameters for fine-tuning"""
        backbone = getattr(self, self.backbone)
        for param in backbone.parameters():
            param.requires_grad = True