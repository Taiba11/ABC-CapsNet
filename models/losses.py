"""
Loss Functions for ABC-CapsNet (Section 3.5).

Implements the Margin Loss used for capsule network classification.
"""

import torch
import torch.nn as nn


class MarginLoss(nn.Module):
    """
    Margin Loss for Capsule Networks (Eq. 7 in the paper).

        L_k = T_k * max(0, m+ - ||v_k||)^2
            + λ * (1 - T_k) * max(0, ||v_k|| - m-)^2

    Where:
        - T_k = 1 if class k is present, 0 otherwise
        - m+ = upper margin (correct class should have ||v_k|| >= m+)
        - m- = lower margin (absent class should have ||v_k|| <= m-)
        - λ  = down-weighting factor for absent classes

    Args:
        m_plus (float): Upper margin threshold. Default: 0.9
        m_minus (float): Lower margin threshold. Default: 0.1
        lambda_val (float): Down-weighting for absent class loss. Default: 0.5
    """

    def __init__(
        self,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_val: float = 0.5,
    ):
        super().__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_val = lambda_val

    def forward(self, v_k, targets):
        """
        Compute margin loss.

        Args:
            v_k: (batch, num_classes, capsule_dim) — digit capsule output vectors.
            targets: (batch,) — integer class labels.

        Returns:
            loss: Scalar margin loss averaged over the batch.
        """
        # Compute capsule lengths (probabilities)
        # ||v_k||: (batch, num_classes)
        v_k_norm = torch.sqrt((v_k ** 2).sum(dim=-1) + 1e-8)

        # One-hot encode targets
        # T_k: (batch, num_classes)
        num_classes = v_k.size(1)
        T_k = torch.zeros(v_k.size(0), num_classes, device=v_k.device)
        T_k.scatter_(1, targets.unsqueeze(1), 1.0)

        # Margin loss computation
        # Present class: T_k * max(0, m+ - ||v_k||)^2
        left = T_k * torch.clamp(self.m_plus - v_k_norm, min=0.0) ** 2

        # Absent class: λ * (1 - T_k) * max(0, ||v_k|| - m-)^2
        right = self.lambda_val * (1.0 - T_k) * torch.clamp(
            v_k_norm - self.m_minus, min=0.0
        ) ** 2

        # Total loss: sum over classes, mean over batch
        loss = (left + right).sum(dim=-1).mean()

        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss: Margin Loss + Cross-Entropy Loss.

    Provides flexibility to use margin loss from capsule networks
    alongside standard cross-entropy for auxiliary supervision.

    Args:
        margin_weight (float): Weight for the margin loss component.
        ce_weight (float): Weight for the cross-entropy loss component.
        m_plus (float): Margin loss upper threshold.
        m_minus (float): Margin loss lower threshold.
        lambda_val (float): Margin loss down-weighting factor.
    """

    def __init__(
        self,
        margin_weight: float = 1.0,
        ce_weight: float = 0.5,
        m_plus: float = 0.9,
        m_minus: float = 0.1,
        lambda_val: float = 0.5,
    ):
        super().__init__()
        self.margin_weight = margin_weight
        self.ce_weight = ce_weight
        self.margin_loss = MarginLoss(m_plus, m_minus, lambda_val)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, v_k, targets):
        """
        Args:
            v_k: (batch, num_classes, capsule_dim) — capsule outputs.
            targets: (batch,) — integer class labels.

        Returns:
            total_loss: Weighted combination of margin and CE losses.
        """
        m_loss = self.margin_loss(v_k, targets)

        # Use capsule lengths as logits for CE
        logits = torch.sqrt((v_k ** 2).sum(dim=-1) + 1e-8)
        ce_loss = self.ce_loss(logits, targets)

        return self.margin_weight * m_loss + self.ce_weight * ce_loss
