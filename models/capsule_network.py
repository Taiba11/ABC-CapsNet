"""
Cascaded Capsule Networks (CN1 & CN2) for ABC-CapsNet (Section 3.4).

CN1: Extracts fundamental capsule-level patterns from attention-refined features.
CN2: Processes CN1 outputs for deeper, more nuanced feature abstraction.

The cascading design enables progressive refinement of audio deepfake features.
"""

import torch
import torch.nn as nn

from .capsule_layers import PrimaryCapsuleLayer, DigitCapsuleLayer, squash


class CapsuleNetwork1(nn.Module):
    """
    Capsule Network 1 (CN1) — Section 3.4.1.

    Architecture:
        Input → Conv Layers → Primary Capsule Layer → Digit Capsule Layer (via Dynamic Routing)

    The first capsule network extracts fundamental patterns and spatial
    hierarchies from the attention-refined feature maps.

    Args:
        in_channels (int): Input channels from the attention module.
        conv_out_channels (int): Output channels of the convolutional layers.
        primary_num_caps (int): Number of primary capsule types.
        primary_cap_dim (int): Dimension of primary capsule vectors.
        primary_kernel (int): Kernel size for primary capsule convolutions.
        primary_stride (int): Stride for primary capsule convolutions.
        digit_num_caps (int): Number of digit capsule types (2 = real/fake).
        digit_cap_dim (int): Dimension of digit capsule vectors.
        routing_iterations (int): Number of dynamic routing iterations.
    """

    def __init__(
        self,
        in_channels: int = 512,
        conv_out_channels: int = 256,
        primary_num_caps: int = 8,
        primary_cap_dim: int = 32,
        primary_kernel: int = 9,
        primary_stride: int = 2,
        digit_num_caps: int = 2,
        digit_cap_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()

        # --- Two convolutional layers preceding capsules ---
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_out_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_out_channels),
            nn.ReLU(inplace=True),
        )

        # --- Primary Capsule Layer ---
        self.primary_capsules = PrimaryCapsuleLayer(
            in_channels=conv_out_channels,
            num_capsules=primary_num_caps,
            capsule_dim=primary_cap_dim,
            kernel_size=primary_kernel,
            stride=primary_stride,
            padding=primary_kernel // 2,
        )

        # We need to compute num_routes dynamically based on input spatial size
        # For 7x7 input with kernel=9, stride=2, padding=4: output = 4x4
        # num_routes = primary_num_caps * H' * W'
        self._primary_num_caps = primary_num_caps
        self._primary_cap_dim = primary_cap_dim
        self._digit_num_caps = digit_num_caps
        self._digit_cap_dim = digit_cap_dim
        self._routing_iterations = routing_iterations

        # DigitCapsuleLayer will be lazily initialized on first forward pass
        self.digit_capsules = None
        self._initialized = False

    def _initialize_digit_caps(self, num_routes: int, device):
        """Lazily initialize digit capsule layer once we know the num_routes."""
        self.digit_capsules = DigitCapsuleLayer(
            num_capsules=self._digit_num_caps,
            num_routes=num_routes,
            in_dim=self._primary_cap_dim,
            out_dim=self._digit_cap_dim,
            routing_iterations=self._routing_iterations,
        ).to(device)
        self._initialized = True

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W) — attention-refined feature maps.

        Returns:
            v_j: (batch, digit_num_caps, digit_cap_dim) — digit capsule outputs.
            primary_caps: (batch, num_total_primary_caps, primary_cap_dim) — for CN2 input.
        """
        # Convolutional layers
        x = self.conv_layers(x)

        # Primary capsules
        primary_caps = self.primary_capsules(x)
        # primary_caps: (batch, num_total_caps, primary_cap_dim)

        # Lazy init digit capsules
        if not self._initialized:
            self._initialize_digit_caps(primary_caps.size(1), x.device)

        # Digit capsules with dynamic routing
        v_j = self.digit_capsules(primary_caps)
        # v_j: (batch, digit_num_caps, digit_cap_dim)

        return v_j, primary_caps


class CapsuleNetwork2(nn.Module):
    """
    Capsule Network 2 (CN2) — Section 3.4.2.

    Architecture:
        Input (CN1 outputs) → Transformation → Secondary Capsule Layer → Digit Capsule Layer

    Processes the output vectors from CN1 through a secondary phase of
    transformation and dynamic routing for deeper feature abstraction.

    Args:
        in_capsule_dim (int): Dimension of input capsule vectors (from CN1 digit caps).
        in_num_capsules (int): Number of input capsules from CN1.
        secondary_num_caps (int): Number of secondary capsule types.
        secondary_cap_dim (int): Dimension of secondary capsule vectors.
        digit_num_caps (int): Number of final digit capsules (2 = real/fake).
        digit_cap_dim (int): Dimension of final digit capsule vectors.
        routing_iterations (int): Number of dynamic routing iterations.
    """

    def __init__(
        self,
        in_capsule_dim: int = 16,
        in_num_capsules: int = 2,
        secondary_num_caps: int = 4,
        secondary_cap_dim: int = 16,
        digit_num_caps: int = 2,
        digit_cap_dim: int = 16,
        routing_iterations: int = 3,
    ):
        super().__init__()

        self.secondary_num_caps = secondary_num_caps
        self.secondary_cap_dim = secondary_cap_dim

        # Transform CN1 digit capsule outputs into secondary capsule format
        self.transform = nn.Sequential(
            nn.Linear(in_capsule_dim, secondary_cap_dim * secondary_num_caps),
            nn.ReLU(inplace=True),
        )

        # Secondary digit capsule layer
        self.digit_capsules = DigitCapsuleLayer(
            num_capsules=digit_num_caps,
            num_routes=secondary_num_caps * in_num_capsules,
            in_dim=secondary_cap_dim,
            out_dim=digit_cap_dim,
            routing_iterations=routing_iterations,
        )

    def forward(self, v_j):
        """
        Args:
            v_j: (batch, num_capsules, capsule_dim) — CN1 digit capsule outputs.

        Returns:
            v_j_prime: (batch, digit_num_caps, digit_cap_dim) — final capsule vectors.
        """
        batch_size = v_j.size(0)

        # Transform each input capsule into multiple secondary capsules
        transformed = self.transform(v_j)
        # transformed: (batch, num_input_caps, secondary_num_caps * secondary_cap_dim)

        # Reshape to capsule format
        transformed = transformed.view(
            batch_size, -1, self.secondary_cap_dim
        )
        # transformed: (batch, num_input_caps * secondary_num_caps, secondary_cap_dim)

        # Apply squash
        transformed = squash(transformed)

        # Dynamic routing
        v_j_prime = self.digit_capsules(transformed)
        # v_j_prime: (batch, digit_num_caps, digit_cap_dim)

        return v_j_prime
