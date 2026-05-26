import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import physics


class DeepAugmentedMUSICDataset(Dataset):
    """
    PyTorch용 DOA 추정 데이터셋 클래스
    x: [N, 2M, T]
    y: [N, 6]  (남는 자리는 np.pi padding)
    """
    def __init__(self, x, y=None):
        if isinstance(x, torch.Tensor):
            self.data = x.float()
        else:
            self.data = torch.from_numpy(x).float()

        if y is not None:
            if isinstance(y, torch.Tensor):
                self.label = y.float()
            else:
                self.label = torch.from_numpy(y).float()
        else:
            self.label = None

    def __getitem__(self, index):
        if self.label is not None:
            return self.data[index], self.label[index]
        return self.data[index]

    def __len__(self):
        return len(self.data)


def sample_thetas_with_min_sep(num, min_sep_deg=8.0):
    """
    최소 각도 간격을 두고 DOA를 샘플링
    """
    min_sep = np.deg2rad(min_sep_deg)
    max_trials = 2000

    for _ in range(max_trials):
        thetas = np.pi * (np.random.rand(num) - 0.5)   # [-pi/2, pi/2]
        thetas = np.sort(thetas)
        if np.all(np.diff(thetas) >= min_sep): # 최소 각도 유지하는 애들만 선택
            return thetas #출력 shape는 64float radinas

    # fallback
    grid = np.linspace(-np.pi / 2 + min_sep, np.pi / 2 - min_sep, 2000) #grid에서 선택
    idx = np.sort(np.random.choice(len(grid), size=num, replace=False))
    thetas = np.sort(grid[idx])

    for i in range(1, len(thetas)):
        if thetas[i] - thetas[i - 1] < min_sep:
            thetas[i] = min(thetas[i - 1] + min_sep, np.pi / 2 - min_sep)

    return thetas #출력 shape는 64float radinas


def create_complete_dataset(
    name,
    size,
    snr,
    snapshots,
    m=8,
    num_sources=[2, 3, 4, 5],
    coherent=False,
    save=True,
    min_sep_deg=8.0,
):
    """
    다양한 개수의 신호원을 가진 데이터를 생성하고 .h5 파일로 저장합니다.
    X: [N, M, T] complex
    Y: [N, 6] float (남는 자리는 pi padding)
    """
    X = np.zeros((size, m, snapshots), dtype=np.complex64)
    Thetas = np.full((size, 6), np.pi, dtype=np.float32)

    for i in tqdm(range(size), desc=f"Generating {name}"):
        num = num_sources[i % len(num_sources)]   # unknown K in {2,3,4,5}
        thetas = sample_thetas_with_min_sep(num, min_sep_deg=min_sep_deg)

        if coherent:
            X_i, _ = physics.construct_coherent_signal(
                thetas, snr, snapshots, m=m
            )
        else:
            X_i, _ = physics.construct_signal(
                thetas, snr, snapshots, m=m
            )

        X[i] = X_i.astype(np.complex64)
        Thetas[i, :num] = thetas.astype(np.float32)

    if save:
        with h5py.File(f"{name}.h5", "w") as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("Y", data=Thetas)

    return X, Thetas


def load_and_preprocess_data(file_path, use_1bit=False):
    """
    .h5 파일을 로드하고 전처리를 수행합니다.
    1) complex -> real/imag split  => [N, 2M, T]
    2) use_1bit=True 이면 sign 양자화
    """
    with h5py.File(file_path, "r") as hf:
        X_raw = np.array(hf.get("X"))   # [N, M, T], complex
        Y = np.array(hf.get("Y"))       # [N, 6]

    X_processed = np.concatenate([X_raw.real, X_raw.imag], axis=1).astype(np.float32)

    if use_1bit:
        X_processed = np.sign(X_processed).astype(np.float32)
        X_processed[X_processed == 0] = 1.0
    else:
        std = np.std(X_processed, axis=(1, 2), keepdims=True) + 1e-8
        X_processed = (X_processed / std).astype(np.float32)

    return X_processed, Y.astype(np.float32)


def get_dataloader(file_path, batch_size=128, shuffle=True, use_1bit=False):
    """
    데이터 로드부터 DataLoader 생성까지 한 번에 처리합니다.
    """
    X, Y = load_and_preprocess_data(file_path, use_1bit=use_1bit)
    dataset = DeepAugmentedMUSICDataset(X, Y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=True,
    )