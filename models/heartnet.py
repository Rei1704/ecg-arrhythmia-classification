"""
models/heartnet.py
HeartNet: CNN + LSTM + Attention hybrid for 5-class ECG classification.

Input: (batch, 1, 180)
CNN -> (batch, 64, 45) -> LSTM -> Attention -> Classifier -> (batch, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _AttentionLayer(nn.Module):
    """Additive attention over LSTM hidden states to get a context vector."""

    def __init__(self, hidden_size):
        super(_AttentionLayer, self).__init__()
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden_size)
        raw     = self.score(lstm_out)              # (batch, seq_len, 1)
        weights = F.softmax(raw, dim=1)             # (batch, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1)   # (batch, hidden_size)
        return context, weights.squeeze(-1)


class HeartNet(nn.Module):
    """Hybrid CNN + LSTM + Attention for 5-class ECG beat classification."""

    def __init__(self, num_classes=5):
        super(HeartNet, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),        # 180 -> 90

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),        # 90 -> 45
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
        )

        self.attention = _AttentionLayer(hidden_size=128)

        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)                     # (batch, 64, 45)
        x = x.permute(0, 2, 1)             # (batch, 45, 64) for LSTM
        lstm_out, _ = self.lstm(x)          # (batch, 45, 128)
        context, _  = self.attention(lstm_out)  # (batch, 128)
        return self.classifier(context)


if __name__ == '__main__':
    model = HeartNet(num_classes=5)
    dummy = torch.randn(16, 1, 180)
    out   = model(dummy)
    print(f"HeartNet  output shape : {out.shape}")
    print(f"HeartNet  total params : {sum(p.numel() for p in model.parameters()):,}")
