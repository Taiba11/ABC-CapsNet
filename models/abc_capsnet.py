"""
ABC-CapsNet: Attention-Based Cascaded Capsule Network for Audio Deepfake Detection.

Full architecture combining:
    1. VGG18 Feature Extraction
    2. Attention Mechanism
    3. Capsule Network 1 (CN1)
    4. Capsule Network 2 (CN2)
    5. Classification via capsule vector norms

Paper: Wani et al., CVPR Workshop 2024
"""

import torch
import torch.nn as nn

from .vgg18 import VGG18FeatureExtractor
from .attention import AttentionModule
from .capsule_network import CapsuleNetwork1, CapsuleNetwork2


class ABCCapsNet(nn.Module):
    """
    ABC-CapsNet (Attention-Based Cascaded Capsule Network).

    Pipeline:
        Mel Spectrogram (224×224×3)
            → VGG18 Feature Extraction
            → Attention Mechanism
            → Capsule Network 1 (CN1)
            → Capsule Network 2 (CN2)
            → Classification (Real / Fake)

    Args:
        num_classes (int): Number of output classes (2 for real/fake).
        pretrained_backbone (bool): Use pretrained VGG16 weights for VGG18.
        freeze_backbone (int): Number of backbone layers to freeze.
        attention_hidden_dim (int): Hidden dimension of attention module.
        attention_dropout (float): Dropout rate in attention module.
        cn1_primary_num_caps (int): Number of CN1 primary capsule types.
        cn1_primary_cap_dim (int): Dimension of CN1 primary capsule vectors.
        cn1_primary_kernel (int): Kernel size for CN1 primary capsules.
        cn1_primary_stride (int): Stride for CN1 primary capsules.
        cn1_digit_cap_dim (int): Dimension of CN1 digit capsule vectors.
        cn2_secondary_num_caps (int): Number of CN2 secondary capsule types.
        cn2_secondary_cap_dim (int): Dimension of CN2 secondary capsule vectors.
        cn2_digit_cap_dim (int): Dimension of CN2 digit capsule vectors.
        routing_iterations (int): Number of dynamic routing iterations.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained_backbone: bool = True,
        freeze_backbone: int = 0,
        attention_hidden_dim: int = 256,
        attention_dropout: float = 0.1,
        cn1_primary_num_caps: int = 8,
        cn1_primary_cap_dim: int = 32,
        cn1_primary_kernel: int = 9,
        cn1_primary_stride: int = 2,
        cn1_digit_cap_dim: int = 16,
        cn2_secondary_num_caps: int = 4,
        cn2_secondary_cap_dim: int = 16,
        cn2_digit_cap_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # --- Stage 1: VGG18 Feature Extraction ---
        self.backbone = VGG18FeatureExtractor(
            pretrained=pretrained_backbone,
            freeze_layers=freeze_backbone,
        )

        # --- Stage 2: Attention Mechanism ---
        self.attention = AttentionModule(
            in_channels=512,
            hidden_dim=attention_hidden_dim,
            dropout=attention_dropout,
        )

        # --- Stage 3: Capsule Network 1 (CN1) ---
        self.capsule_net_1 = CapsuleNetwork1(
            in_channels=512,
            conv_out_channels=256,
            primary_num_caps=cn1_primary_num_caps,
            primary_cap_dim=cn1_primary_cap_dim,
            primary_kernel=cn1_primary_kernel,
            primary_stride=cn1_primary_stride,
            digit_num_caps=num_classes,
            digit_cap_dim=cn1_digit_cap_dim,
            routing_iterations=routing_iterations,
        )

        # --- Stage 4: Capsule Network 2 (CN2) ---
        self.capsule_net_2 = CapsuleNetwork2(
            in_capsule_dim=cn1_digit_cap_dim,
            in_num_capsules=num_classes,
            secondary_num_caps=cn2_secondary_num_caps,
            secondary_cap_dim=cn2_secondary_cap_dim,
            digit_num_caps=num_classes,
            digit_cap_dim=cn2_digit_cap_dim,
            routing_iterations=routing_iterations,
        )

    def forward(self, x):
        """
        Full forward pass through ABC-CapsNet.

        Args:
            x: (batch, 3, 224, 224) — Mel spectrogram images.

        Returns:
            v_j_prime: (batch, num_classes, digit_cap_dim) — final capsule vectors.
                       The L2 norm of each capsule vector represents the probability
                       of the corresponding class being present.
        """
        # Stage 1: Feature extraction
        features = self.backbone(x)
        # features: (batch, 512, 7, 7)

        # Stage 2: Attention refinement
        attended = self.attention(features)
        # attended: (batch, 512, 7, 7)

        # Stage 3: Capsule Network 1
        v_j, _ = self.capsule_net_1(attended)
        # v_j: (batch, num_classes, cn1_digit_cap_dim)

        # Stage 4: Capsule Network 2
        v_j_prime = self.capsule_net_2(v_j)
        # v_j_prime: (batch, num_classes, cn2_digit_cap_dim)

        return v_j_prime

    def predict(self, x):
        """
        Predict class labels and confidence scores.

        Args:
            x: (batch, 3, 224, 224) — Mel spectrogram images.

        Returns:
            predictions: (batch,) — predicted class indices.
            confidences: (batch, num_classes) — capsule norms as confidence.
        """
        v_j_prime = self.forward(x)

        # Capsule norms as class probabilities
        confidences = torch.sqrt((v_j_prime ** 2).sum(dim=-1) + 1e-8)
        # confidences: (batch, num_classes)

        predictions = confidences.argmax(dim=1)
        # predictions: (batch,)

        return predictions, confidences

    def get_num_params(self):
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
