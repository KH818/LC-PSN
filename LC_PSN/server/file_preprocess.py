import io
# import h5py
import numpy as np
import torch

def npy_complex_to_model_input(file_bytes: bytes, use_1bit: bool = True) -> torch.Tensor:
    """
    .npy 파일 bytes를 읽어서 LC-PSN 입력 tensor로 변환.

    입력 .npy:
      X: [M, T] complex

    출력:
      x: [1, 2M, T] float32 torch.Tensor
    """
    X_raw = np.load(io.BytesIO(file_bytes))  # [M, T], complex

    if X_raw.ndim != 2:
        raise ValueError(f"Input npy must have shape [M, T], got {X_raw.shape}")

    if not np.iscomplexobj(X_raw):
        raise ValueError("Input npy must contain complex values")

    M, T = X_raw.shape

    if M != 8 or T != 200:
        raise ValueError(f"Expected shape [8, 200], got {X_raw.shape}")

    X_raw = X_raw[None, :, :]  # [1, M, T]

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