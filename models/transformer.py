"""
models/transformer.py
Transformer Encoder for 5-class ECG arrhythmia classification.

Input: (batch, 1, 180)
Conv1d embedding -> Positional Encoding -> 3x TransformerEncoderLayer
-> Global Avg Pool -> Classifier -> (batch, 5)
"""

import math
import torch
import torch.nn as nn


class _PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to the input sequence."""

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.dropout(x + self.pe[:, :x.size(1), :])


class ECGTransformer(nn.Module):
    """Transformer Encoder for 5-class ECG beat classification."""

    def __init__(self, num_classes=5, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=256, dropout=0.1):
        super(ECGTransformer, self).__init__()

        self.embedding    = nn.Conv1d(1, d_model, kernel_size=1)
        self.pos_encoding = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.embedding(x)       # (batch, d_model, 180)
        x = x.permute(0, 2, 1)     # (batch, 180, d_model)
        x = self.pos_encoding(x)    # add positional encoding
        x = self.transformer(x)     # (batch, 180, d_model)
        x = x.mean(dim=1)           # global average pool -> (batch, d_model)
        return self.classifier(x)


if __name__ == '__main__':
    model = ECGTransformer(num_classes=5)
    dummy = torch.randn(16, 1, 180)
    out   = model(dummy)
    print(f"ECGTransformer  output shape : {out.shape}")
    print(f"ECGTransformer  total params : {sum(p.numel() for p in model.parameters()):,}")
