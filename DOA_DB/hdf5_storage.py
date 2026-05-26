import io
import os
from datetime import datetime
from typing import Any, Optional

import h5py
import numpy as np


RAW_STORAGE_DIR = "raw_storage"
SPECTRUM_STORAGE_DIR = "spectrum_storage"

os.makedirs(RAW_STORAGE_DIR, exist_ok=True)
os.makedirs(SPECTRUM_STORAGE_DIR, exist_ok=True)


#POST /infer-npy로 들어온 .npy raw signal을 HDF5 파일로 저장
def save_raw_npy_as_hdf5(
    raw_data_id: str,
    file_bytes: bytes,
    sensor_id: str,
    received_at: Optional[datetime] = None,
):
    X_raw = np.load(io.BytesIO(file_bytes))

    if X_raw.ndim != 2:
        raise ValueError(f"Input npy must have shape [M, T], got {X_raw.shape}")

    if not np.iscomplexobj(X_raw):
        raise ValueError("Input npy must contain complex values")

    antenna_count = int(X_raw.shape[0])
    snapshot_count = int(X_raw.shape[1])

    if antenna_count != 8 or snapshot_count != 200:
        raise ValueError(f"Expected shape [8, 200], got {X_raw.shape}")

    file_path = os.path.join(RAW_STORAGE_DIR, f"{raw_data_id}.h5")

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("X", data=X_raw.astype(np.complex64))

        hf.attrs["raw_data_id"] = raw_data_id
        hf.attrs["sensor_id"] = sensor_id
        hf.attrs["antenna_count"] = antenna_count
        hf.attrs["snapshot_count"] = snapshot_count
        hf.attrs["raw_format"] = "h5"
        hf.attrs["source_format"] = "npy"

        if received_at is not None:
            hf.attrs["received_at"] = received_at.isoformat()

    return file_path, "h5", antenna_count, snapshot_count


#스펙트럼 저장 함수
def save_spectrum_as_hdf5(raw_data_id: str, spectrum: Any):
    if spectrum is None:
        return None

    spectrum_array = np.array(spectrum, dtype=np.float32)

    if spectrum_array.ndim == 0:
        raise ValueError("spectrum must be an array-like value")

    spectrum_path = os.path.join(
        SPECTRUM_STORAGE_DIR,
        f"{raw_data_id}_spectrum.h5"
    )

    with h5py.File(spectrum_path, "w") as hf:
        hf.create_dataset("spectrum", data=spectrum_array)
        hf.attrs["raw_data_id"] = raw_data_id
        hf.attrs["spectrum_shape"] = str(list(spectrum_array.shape))

    return spectrum_path