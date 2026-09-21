"""Wrap baseline input arrays and labels as a PyTorch Dataset."""

import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, x, y=None):
        self.data = torch.from_numpy(x).float()
        if y is not None:
            self.label = torch.from_numpy(y).float()
        else:
            self.label = None

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)
