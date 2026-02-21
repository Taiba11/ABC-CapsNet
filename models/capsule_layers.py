"""
Capsule Layer Primitives for ABC-CapsNet.

Implements:
    - squash activation function
    - PrimaryCapsuleLayer
    - DigitCapsuleLayer with dynamic routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(tensor, dim=-1):
    """
    Squash activation function (Eq. 6 in the paper).

    Shrinks the vector length to [0, 1] while preserving direction.

        squash(v) = (||v||^2 / (1 + ||v||^2)) * (v / ||v||)

    Args:
        tensor: Input tensor.
        dim: Dimension along which to compute the norm.

    Returns:
        Squashed tensor with the same shape.
    """
    squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
    norm = torch.sqrt(squared_norm + 1e-8)
    scale = squared_norm / (1.0 + squared_norm)
    return scale * tensor / norm


class PrimaryCapsuleLayer(nn.Module):
    """
    Primary Capsule Layer.

    Converts conventional CNN feature maps into capsule format by
    reshaping conv outputs into groups of vectors (capsules).

    Each capsule is a group of neurons whose activity vector represents
    the instantiation parameters of a specific entity type.

    Args:
        in_channels (int): Number of input channels from preceding conv layer.
        num_capsules (int): Number of capsule types.
        capsule_dim (int): Dimension of each capsule's output vector.
        kernel_size (int): Convolution kernel size.
        stride (int): Convolution stride.
        padding (int): Convolution padding.
    """

    def __init__(
        self,
        in_channels: int,
        num_capsules: int,
        capsule_dim: int,
        kernel_size: int = 9,
        stride: int = 2,
        padding: int = 0,
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim

        # Each capsule type has its own convolutional layer
        self.capsules = nn.ModuleList([
            nn.Conv2d(
                in_channels,
                capsule_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            for _ in range(num_capsules)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, H, W) — feature maps from conv layers.

        Returns:
            (batch, num_capsules * H' * W', capsule_dim) — primary capsule outputs.
        """
        outputs = [capsule(x) for capsule in self.capsules]
        # Each output: (batch, capsule_dim, H', W')
        outputs = torch.stack(outputs, dim=1)
        # (batch, num_capsules, capsule_dim, H', W')
        batch_size = outputs.size(0)
        # Reshape: merge spatial dims with capsule count
        outputs = outputs.permute(0, 1, 3, 4, 2).contiguous()
        # (batch, num_capsules, H', W', capsule_dim)
        outputs = outputs.view(batch_size, -1, self.capsule_dim)
        # (batch, num_total_capsules, capsule_dim)
        return squash(outputs)


class DigitCapsuleLayer(nn.Module):
    """
    Digit Capsule Layer with Dynamic Routing (Algorithm 1 in the paper).

    Implements the higher-level capsule layer that uses dynamic routing
    to determine coupling coefficients between lower-level and higher-level
    capsules.

    Args:
        num_capsules (int): Number of output capsule types (e.g., 2 for real/fake).
        num_routes (int): Number of input capsules (from primary layer).
        in_dim (int): Dimension of input capsule vectors.
        out_dim (int): Dimension of output capsule vectors.
        routing_iterations (int): Number of dynamic routing iterations.
    """

    def __init__(
        self,
        num_capsules: int,
        num_routes: int,
        in_dim: int,
        out_dim: int,
        routing_iterations: int = 3,
    ):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.routing_iterations = routing_iterations

        # Transformation matrices W_ij: (1, num_routes, num_capsules, out_dim, in_dim)
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, out_dim, in_dim) * 0.01
        )

    def forward(self, x):
        """
        Dynamic routing between capsules.

        Args:
            x: (batch, num_routes, in_dim) — input capsule vectors.

        Returns:
            (batch, num_capsules, out_dim) — output capsule vectors.
        """
        batch_size = x.size(0)

        # Expand input for matrix multiplication
        # x: (batch, num_routes, 1, in_dim, 1)
        x = x.unsqueeze(2).unsqueeze(4)

        # Compute prediction vectors u_hat = W @ x
        # W: (1, num_routes, num_capsules, out_dim, in_dim)
        # x: (batch, num_routes, 1, in_dim, 1)
        W = self.W.expand(batch_size, -1, -1, -1, -1)
        u_hat = torch.matmul(W, x).squeeze(-1)
        # u_hat: (batch, num_routes, num_capsules, out_dim)

        # Initialize log-prior probabilities b_ij = 0
        b_ij = torch.zeros(
            batch_size, self.num_routes, self.num_capsules, 1,
            device=x.device
        )

        # Dynamic routing iterations
        for iteration in range(self.routing_iterations):
            # Step 1: Compute coupling coefficients c_ij via softmax (Eq. in Alg. 1)
            c_ij = F.softmax(b_ij, dim=2)
            # c_ij: (batch, num_routes, num_capsules, 1)

            # Step 2: Compute weighted sum s_j = sum_i(c_ij * u_hat_ij)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            # s_j: (batch, 1, num_capsules, out_dim)

            # Step 3: Squash to get output v_j
            v_j = squash(s_j, dim=-1)
            # v_j: (batch, 1, num_capsules, out_dim)

            # Step 4: Update b_ij (agreement)
            if iteration < self.routing_iterations - 1:
                # agreement = u_hat . v_j
                agreement = (u_hat * v_j).sum(dim=-1, keepdim=True)
                # agreement: (batch, num_routes, num_capsules, 1)
                b_ij = b_ij + agreement

        # Remove the keepdim=1 route dimension
        v_j = v_j.squeeze(1)
        # v_j: (batch, num_capsules, out_dim)

        return v_j
