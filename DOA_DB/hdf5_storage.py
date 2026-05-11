import os
import uuid
import json
import shutil
import h5py
import numpy as np
from fastapi import UploadFile


RAW_STORAGE_DIR = "raw_storage"
os.makedirs(RAW_STORAGE_DIR, exist_ok=True)


def save_uploaded_h5_file(file: UploadFile):
    raw_id = str(uuid.uuid4())

    ext = os.path.splitext(file.filename)[1]
    if ext == "":
        ext = ".h5"

    file_path = os.path.join(RAW_STORAGE_DIR, f"{raw_id}{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    antenna_count = None
    snapshot_count = None

    try:
        with h5py.File(file_path, "r") as hf:
            if "X" in hf:
                x = np.array(hf["X"])
                if x.ndim == 3:
                    antenna_count = int(x.shape[1])
                    snapshot_count = int(x.shape[2])
                elif x.ndim == 2:
                    antenna_count = int(x.shape[0])
                    snapshot_count = int(x.shape[1])

            elif "iq_real" in hf and "iq_imag" in hf:
                real = np.array(hf["iq_real"])
                antenna_count = int(real.shape[0])
                snapshot_count = int(real.shape[1])

            antenna_count = int(hf.attrs.get("antenna_count", antenna_count or -1))
            snapshot_count = int(hf.attrs.get("snapshot_count", snapshot_count or -1))

    except Exception:
        antenna_count = -1
        snapshot_count = -1

    return raw_id, file_path, antenna_count, snapshot_count


def save_iq_json_as_hdf5(
    iq_real_json: str,
    iq_imag_json: str,
    sample_rate: float | None = None,
    center_frequency: float | None = None,
):
    raw_id = str(uuid.uuid4())
    file_path = os.path.join(RAW_STORAGE_DIR, f"{raw_id}.h5")

    real_array = np.array(json.loads(iq_real_json), dtype=np.float32)
    imag_array = np.array(json.loads(iq_imag_json), dtype=np.float32)

    if real_array.shape != imag_array.shape:
        raise ValueError("iq_real and iq_imag must have same shape")

    if real_array.ndim != 2:
        raise ValueError("iq_real and iq_imag must be 2D arrays: [antenna_count, snapshot_count]")

    antenna_count = int(real_array.shape[0])
    snapshot_count = int(real_array.shape[1])

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("iq_real", data=real_array)
        hf.create_dataset("iq_imag", data=imag_array)
        hf.attrs["antenna_count"] = antenna_count
        hf.attrs["snapshot_count"] = snapshot_count

        if sample_rate is not None:
            hf.attrs["sample_rate"] = sample_rate

        if center_frequency is not None:
            hf.attrs["center_frequency"] = center_frequency

    return raw_id, file_path, antenna_count, snapshot_count