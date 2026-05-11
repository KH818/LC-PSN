import uuid
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from hdf5_storage import save_uploaded_h5_file, save_iq_json_as_hdf5
from influx_client import (
    save_raw_metadata,
    save_inference_result,
    get_latest_inference_result,
    get_recent_inference_results,
    get_results_by_raw_id,
    get_raw_signals,
    get_history_summary,
)


app = FastAPI(
    title="DOA DB Server",
    description="Raw IQ HDF5 storage + DOA inference result storage with InfluxDB",
    version="1.0.0",
)


class InferenceResultInput(BaseModel):
    raw_id: str
    estimated_k: int
    doa_angles_deg: List[float]
    snr_estimate: Optional[float] = None
    confidence: Optional[float] = None


@app.get("/")
def root():
    return {
        "message": "DOA DB Server is running",
        "role": "Raw signal storage + inference result storage",
    }


@app.post("/raw-signals/upload-h5")
async def upload_h5_raw_signal(
    file: UploadFile = File(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
):
    try:
        raw_id, file_path, antenna_count, snapshot_count = save_uploaded_h5_file(file)

        save_raw_metadata(
            raw_id=raw_id,
            file_path=file_path,
            antenna_count=antenna_count,
            snapshot_count=snapshot_count,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            description=description,
        )

        return {
            "raw_id": raw_id,
            "file_path": file_path,
            "antenna_count": antenna_count,
            "snapshot_count": snapshot_count,
            "message": "Raw HDF5 signal uploaded and metadata saved",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/raw-signals/create-hdf5")
def create_hdf5_from_iq_json(
    iq_real: str = Form(...),
    iq_imag: str = Form(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
):
    try:
        raw_id, file_path, antenna_count, snapshot_count = save_iq_json_as_hdf5(
            iq_real_json=iq_real,
            iq_imag_json=iq_imag,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
        )

        save_raw_metadata(
            raw_id=raw_id,
            file_path=file_path,
            antenna_count=antenna_count,
            snapshot_count=snapshot_count,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            description=description,
        )

        return {
            "raw_id": raw_id,
            "file_path": file_path,
            "antenna_count": antenna_count,
            "snapshot_count": snapshot_count,
            "message": "Raw IQ signal saved as HDF5 and metadata saved",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/inference-results/save")
def save_model_inference_result(data: InferenceResultInput):
    try:
        result_id = str(uuid.uuid4())

        save_inference_result(
            raw_id=data.raw_id,
            result_id=result_id,
            estimated_k=data.estimated_k,
            doa_angles_deg=data.doa_angles_deg,
            snr_estimate=data.snr_estimate,
            confidence=data.confidence,
        )

        return {
            "result_id": result_id,
            "raw_id": data.raw_id,
            "estimated_k": data.estimated_k,
            "doa_angles_deg": data.doa_angles_deg,
            "snr_estimate": data.snr_estimate,
            "confidence": data.confidence,
            "message": "Inference result saved",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/raw-signals/recent")
def read_recent_raw_signals(limit: int = 20):
    return get_raw_signals(limit=limit)


@app.get("/inference-results/latest")
def read_latest_inference_result():
    result = get_latest_inference_result()

    if result is None:
        raise HTTPException(status_code=404, detail="No inference result found")

    return result


@app.get("/inference-results/recent")
def read_recent_inference_results(limit: int = 20):
    return get_recent_inference_results(limit=limit)


@app.get("/raw-signals/{raw_id}/results")
def read_results_by_raw_id(raw_id: str):
    return get_results_by_raw_id(raw_id=raw_id)


@app.get("/history/summary")
def read_history_summary(hours: int = 24):
    return get_history_summary(hours=hours)