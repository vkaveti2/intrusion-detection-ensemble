"""
cnn_arch.py — Shared 1D-CNN architecture definition.

This module is the single source of truth for the IDS1DCNN network graph.
Both train_cnn.py (for training) and ensemble.py (for inference) import
from here, ensuring the architecture reconstructed at load-time is
byte-for-byte identical to the one saved during training.

Network design rationale:
  - 1D-CNN treats the 77-feature vector as a short 1D signal.
  - Conv → BatchNorm → ReLU stacks extract local feature co-occurrences
    without assuming temporal ordering (unlike RNNs).
  - AdaptiveAvgPool collapses spatial dim so the classifier head is
    independent of input length — easy to adapt if feature count changes.
  - WeightedRandomSampler in training handles the severe class imbalance
    in CICIDS-2017 (BENIGN >> attack samples).
"""

import torch.nn as nn


class IDS1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for multi-class IDS traffic classification.

    Args:
        n_features  (int): Number of input features (77 for CIC-IDS 2017).
        n_classes   (int): Number of output classes (15 for CIC-IDS 2017).
        hidden_dims (list[int]): Output channels for each Conv1d block.
        fc_dims     (list[int]): Hidden sizes for the FC classifier head.
        dropout     (float): Dropout probability applied after each FC layer.
    """

    def __init__(self, n_features: int, n_classes: int,
                 hidden_dims: list, fc_dims: list, dropout: float = 0.3):
        super().__init__()

        # ── Convolutional feature extractor ───────────────────────────────
        conv_blocks = []
        in_ch = 1
        for out_ch in hidden_dims:
            conv_blocks += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        # Global average pooling collapses the spatial dimension to 1
        conv_blocks.append(nn.AdaptiveAvgPool1d(1))
        self.conv = nn.Sequential(*conv_blocks)

        # ── Fully-connected classifier head ───────────────────────────────
        fc_blocks = []
        in_dim = hidden_dims[-1]
        for out_dim in fc_dims:
            fc_blocks += [
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = out_dim
        fc_blocks.append(nn.Linear(in_dim, n_classes))
        self.fc = nn.Sequential(*fc_blocks)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, n_features).
        Returns:
            logits: Tensor of shape (batch, n_classes). Raw (un-softmaxed) logits.
        """
        # Add channel dim: (batch, n_features) → (batch, 1, n_features)
        x = x.unsqueeze(1)
        # Conv stack + pool: → (batch, hidden_dims[-1], 1)
        x = self.conv(x).squeeze(-1)    # → (batch, hidden_dims[-1])
        return self.fc(x)               # → (batch, n_classes)
