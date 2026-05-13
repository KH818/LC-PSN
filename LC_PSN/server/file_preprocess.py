import h5py
import numpy as np
import torch


def load_first_sample_from_h5(file_path: str, use_1bit: bool = True) -> torch.Tensor:
    """
    h5 파일에서 첫 번째 X 샘플만 읽어서 LC-PSN 입력 tensor로 변환.

    입력 h5:
      X: [N, M, T] complex

    출력:
      x: [1, 2M, T] float32 torch.Tensor
    """
    with h5py.File(file_path, "r") as hf:
        if "X" not in hf:
            raise ValueError("h5 file must contain dataset 'X'")

        X_raw = np.array(hf["X"][0:1])  # [1, M, T], complex

    X_processed = np.concatenate(
        [X_raw.real, X_raw.imag],
        axis=1
    ).astype(np.float32)  # [1, 2M, T]

    if use_1bit:
        X_processed = np.sign(X_processed).astype(np.float32)
        X_processed[X_processed == 0] = 1.0
    else:
        std = np.std(X_processed, axis=(1, 2), keepdims=True) + 1e-8
        X_processed = (X_processed / std).astype(np.float32)

    return torch.from_numpy(X_processed).float()