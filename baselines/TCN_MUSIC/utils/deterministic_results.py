"""Configure NumPy and PyTorch for repeatable baseline training runs."""

import torch
import numpy as np


def deterministic_results(seed=None):
    if seed is None:
        seed = 42
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
