"""
VGG18 Feature Extractor for ABC-CapsNet.

VGG18 consists of 16 convolutional layers and 2 fully-connected layers
(the final FC classifier is removed for feature extraction).
We use pretrained ImageNet weights and treat Mel spectrograms as RGB images.
"""

import torch
import torch.nn as nn
from torchvision import models


def _make_vgg18_layers():
    """
    Build VGG18 layer configuration.

    VGG18 = VGG16 + 2 extra conv layers (one in block 3 and one in block 4).
    Configuration: [64,64,M, 128,128,M, 256,256,256,256,M, 512,512,512,512,M, 512,512,512,512,M]
    Total: 18 weight layers (16 conv + 2 FC used only if needed).
    """
    cfg = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M",
    ]

    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(inplace=True),
            ])
            in_channels = v

    return nn.Sequential(*layers)


class VGG18FeatureExtractor(nn.Module):
    """
    VGG18-based feature extractor.

    Takes 224×224×3 Mel spectrogram images and produces a feature tensor.
    Uses pretrained VGG16 weights for the overlapping layers and randomly
    initializes the two additional conv layers.

    Args:
        pretrained (bool): Whether to initialize from pretrained VGG16 weights.
        freeze_layers (int): Number of initial layers to freeze.
    """

    def __init__(self, pretrained: bool = True, freeze_layers: int = 0):
        super().__init__()

        self.features = _make_vgg18_layers()

        # Transfer weights from pretrained VGG16 for matching layers
        if pretrained:
            self._load_pretrained_weights()

        # Freeze early layers if requested
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

        # Adaptive pooling to fixed spatial size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def _load_pretrained_weights(self):
        """Transfer pretrained VGG16 weights to matching VGG18 layers."""
        vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
        vgg16_state = vgg16.features.state_dict()
        vgg18_state = self.features.state_dict()

        # Map VGG16 layers to VGG18 (they share the first layers)
        vgg16_keys = list(vgg16_state.keys())
        vgg18_keys = list(vgg18_state.keys())

        # Copy weights for matching layers
        for i, key16 in enumerate(vgg16_keys):
            if i < len(vgg18_keys):
                if vgg16_state[key16].shape == vgg18_state[vgg18_keys[i]].shape:
                    vgg18_state[vgg18_keys[i]] = vgg16_state[key16]

        self.features.load_state_dict(vgg18_state, strict=False)

    def _freeze_layers(self, n: int):
        """Freeze the first n layers of the feature extractor."""
        for i, param in enumerate(self.features.parameters()):
            if i < n:
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 224, 224) — Mel spectrogram images.

        Returns:
            (batch, 512, 7, 7) — extracted feature maps.
        """
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x
