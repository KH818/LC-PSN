"""Add sinusoidal time-position features before Transformer encoding."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_len: int = 5000,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 ):
        super().__init__()
        self.device = device
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x += self.pe[:x.size(0)].to(self.device)
        return x
