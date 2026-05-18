import json
import os
import shutil
import uuid
from typing import Any, Optional

import h5py
import numpy as np
from fastapi import UploadFile


RAW_STORAGE_DIR = "raw_storage"
SPECTRUM_STORAGE_DIR = "spectrum_storage"

os.makedirs(RAW_STORAGE_DIR, exist_ok=True)
os.makedirs(SPECTRUM_STORAGE_DIR, exist_ok=True)

#파일 확장자 추출
def _get_file_extension(filename: Optional[str], default_ext: str = ".h5") -> str:
    if not filename:
        return default_ext

    ext = os.path.splitext(filename)[1].lower()
    return ext if ext else default_ext

# .h5 파일 안의 데이터 크기를 읽어서 antenna 개수와 snapshot 개수 추정
def _detect_hdf5_shape(file_path: str):
    antenna_count = -1
    snapshot_count = -1

    try:
        with h5py.File(file_path, "r") as hf:
            if "X" in hf:
                x = np.array(hf["X"]) #dataset을 numpy 배열로 변환
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

            antenna_count = int(hf.attrs.get("antenna_count", antenna_count))
            snapshot_count = int(hf.attrs.get("snapshot_count", snapshot_count))

    except Exception:
        antenna_count = -1
        snapshot_count = -1

    return antenna_count, snapshot_count

#fastAPI로 업로드된 파일 저장
def save_uploaded_h5_file(file: UploadFile):
    id = str(uuid.uuid4())

    ext = _get_file_extension(file.filename, default_ext=".h5")
    raw_format = ext.replace(".", "")

    file_path = os.path.join(RAW_STORAGE_DIR, f"{id}{ext}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    antenna_count, snapshot_count = _detect_hdf5_shape(file_path)

#반환값은 FastAPI에서 InfluxDB에 메타데이터 저장 시 사용됨
    return id, file_path, raw_format, antenna_count, snapshot_count

#JSON 형태로 들어온 IQ 데이터를 HDF5 파일로 저장
#IQ 배열을 python 배열로 바꾸고 numpy 배열로 변환
def save_iq_json_as_hdf5(
    iq_real_json: str,
    iq_imag_json: str,
    sample_rate: Optional[float] = None,
    center_frequency: Optional[float] = None,
):
    id = str(uuid.uuid4())
    raw_format = "h5"
    file_path = os.path.join(RAW_STORAGE_DIR, f"{id}.h5")

    real_array = np.array(json.loads(iq_real_json), dtype=np.float32)
    imag_array = np.array(json.loads(iq_imag_json), dtype=np.float32)

    if real_array.shape != imag_array.shape:
        raise ValueError("iq_real and iq_imag must have same shape")

    if real_array.ndim != 2:
        raise ValueError(
            "iq_real and iq_imag must be 2D arrays: [antenna_count, snapshot_count]"
        )

    antenna_count = int(real_array.shape[0])
    snapshot_count = int(real_array.shape[1])

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("iq_real", data=real_array)
        hf.create_dataset("iq_imag", data=imag_array)

        hf.attrs["id"] = id
        hf.attrs["antenna_count"] = antenna_count
        hf.attrs["snapshot_count"] = snapshot_count
        hf.attrs["raw_format"] = raw_format

        if sample_rate is not None:
            hf.attrs["sample_rate"] = sample_rate

        if center_frequency is not None:
            hf.attrs["center_frequency"] = center_frequency

    return id, file_path, raw_format, antenna_count, snapshot_count

#스펙트럼 저장 함수
def save_spectrum_as_hdf5(id: str, spectrum: Any):
    if spectrum is None:
        return None

    spectrum_array = np.array(spectrum, dtype=np.float32)

    if spectrum_array.ndim == 0:
        raise ValueError("spectrum must be an array-like value")

    spectrum_path = os.path.join(SPECTRUM_STORAGE_DIR, f"{id}_spectrum.h5")

    with h5py.File(spectrum_path, "w") as hf:
        hf.create_dataset("spectrum", data=spectrum_array)
        hf.attrs["id"] = id
        hf.attrs["spectrum_shape"] = str(list(spectrum_array.shape))

    return spectrum_path