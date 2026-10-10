"""
models/cnn.py
Plain 1D CNN baseline for 5-class ECG arrhythmia classification.

Input: (batch, 1, 180)
3 conv blocks -> Flatten -> Linear(2816, 256) -> Linear(256, 5)
"""

import torch
import torch.nn as nn


class PlainCNN(nn.Module):
    """Baseline 1D CNN for 5-class ECG beat classification."""

    def __init__(self, num_classes=5):
        super(PlainCNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),    # 180 -> 90
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),    # 90 -> 45
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),    # 45 -> 22
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 22, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)


if __name__ == '__main__':
    model = PlainCNN(num_classes=5)
    dummy = torch.randn(16, 1, 180)
    out   = model(dummy)
    print(f"PlainCNN  output shape : {out.shape}")
    print(f"PlainCNN  total params : {sum(p.numel() for p in model.parameters()):,}")
