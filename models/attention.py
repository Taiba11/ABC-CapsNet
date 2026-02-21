"""
Attention Mechanism for ABC-CapsNet (Section 3.3).

Implements a channel-spatial attention module that assigns learned weights
to each feature vector from the VGG18 backbone, amplifying the most
discriminative features for deepfake detection.

    F' = sum(w_i * f_i)
    w_i = softmax(s(f_i))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModule(nn.Module):
    """
    Attention mechanism applied after VGG18 feature extraction.

    Computes a scalar attention score for each spatial feature vector
    using a trainable fully-connected attention layer, then produces a
    weighted combination of the feature vectors.

    This module operates on the channel dimension, treating each spatial
    position's channel vector as a feature vector f_i.

    Args:
        in_channels (int): Number of input channels (e.g., 512 from VGG18).
        hidden_dim (int): Hidden dimension of the attention scoring network.
        dropout (float): Dropout rate in the attention scoring network.
    """

    def __init__(
        self,
        in_channels: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Attention scoring network: s(f_i)
        self.attention_fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Optional: channel attention (SE-style) for global context
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, C, H, W) — feature maps from VGG18.

        Returns:
            attended: (batch, C, H, W) — attention-refined feature maps.
        """
        batch, C, H, W = x.shape

        # --- Channel Attention (global context) ---
        channel_weights = self.channel_attention(x)
        # channel_weights: (batch, C)
        x = x * channel_weights.unsqueeze(-1).unsqueeze(-1)

        # --- Spatial Attention (Eq. 3 & 4 in the paper) ---
        # Reshape to (batch, H*W, C) — treat each spatial position as f_i
        features = x.permute(0, 2, 3, 1).contiguous().view(batch, H * W, C)

        # Compute attention scores: s(f_i)
        scores = self.attention_fc(features)
        # scores: (batch, H*W, 1)

        # Softmax over spatial positions (Eq. 4)
        weights = F.softmax(scores, dim=1)
        # weights: (batch, H*W, 1)

        # Apply weights (Eq. 3): F' = sum(w_i * f_i)
        # Instead of collapsing to a single vector, we scale each spatial feature
        weighted_features = features * weights
        # weighted_features: (batch, H*W, C)

        # Reshape back to spatial format
        attended = weighted_features.view(batch, H, W, C).permute(0, 3, 1, 2)
        # attended: (batch, C, H, W)

        return attended
