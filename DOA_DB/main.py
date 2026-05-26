from datetime import datetime
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from hdf5_storage import (
    save_iq_json_as_hdf5,
    save_spectrum_as_hdf5,
    save_uploaded_h5_file,
)
from influx_client import (
    get_history_summary,
    get_latest_inference_result,
    get_raw_signals,
    get_recent_inference_results,
    get_results_by_id,
    get_results_by_time_range,
    save_inference_result,
    save_raw_metadata,
)

#fastAPI 객체 생성
app = FastAPI(
    title="DOA DB Server",
    description="Raw IQ HDF5 storage + DOA inference result storage with InfluxDB",
    version="2.1.0",
)

#모델 추론 결과를 저장할 때 받을 JSON 형식 정의
class InferenceResultInput(BaseModel):
    id: str
    raw_format: str

    input_timestamp: datetime
    output_timestamp: datetime

    k_estimate: int
    doa: List[float]

    spectrum: Optional[Any] = None

    snr_estimate: Optional[float] = None
    confidence: Optional[float] = None

#브라우저에서 get 요청시
@app.get("/")
def root():
    return {
        "message": "DOA DB Server is running",
        "role": "Raw signal storage + inference result storage",
        "version": "2.1.0",
    }

#h5 파일 업로드 API
@app.post("/raw-signals/upload-h5")
async def upload_h5_raw_signal(
    file: UploadFile = File(...),
    input_timestamp: datetime = Form(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
):
    try:
        id, file_path, raw_format, antenna_count, snapshot_count = save_uploaded_h5_file(file)

        save_raw_metadata(
            id=id,
            file_path=file_path,
            raw_format=raw_format,
            input_timestamp=input_timestamp,
            antenna_count=antenna_count,
            snapshot_count=snapshot_count,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            description=description,
        )

        return {
            "id": id,
            "file_path": file_path,
            "raw_format": raw_format,
            "input_timestamp": input_timestamp.isoformat(),
            "antenna_count": antenna_count,
            "snapshot_count": snapshot_count,
            "message": "Raw signal uploaded and metadata saved",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#JSON IQ -> HDF5 생성 API
@app.post("/raw-signals/create-hdf5")
def create_hdf5_from_iq_json(
    iq_real: str = Form(...),
    iq_imag: str = Form(...),
    input_timestamp: datetime = Form(...),
    sample_rate: Optional[float] = Form(None),
    center_frequency: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
):
    try:
        id, file_path, raw_format, antenna_count, snapshot_count = save_iq_json_as_hdf5(
            iq_real_json=iq_real,
            iq_imag_json=iq_imag,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
        )
#InfluxDB에 raw metadata 저장
        save_raw_metadata(
            id=id,
            file_path=file_path,
            raw_format=raw_format,
            input_timestamp=input_timestamp,
            antenna_count=antenna_count,
            snapshot_count=snapshot_count,
            sample_rate=sample_rate,
            center_frequency=center_frequency,
            description=description,
        )

        return {
            "id": id,
            "file_path": file_path,
            "raw_format": raw_format,
            "input_timestamp": input_timestamp.isoformat(),
            "antenna_count": antenna_count,
            "snapshot_count": snapshot_count,
            "message": "Raw IQ signal saved as HDF5 and metadata saved",
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

#추론 결과 저장 API
@app.post("/inference-results/save")
def save_model_inference_result(data: InferenceResultInput):
    try:
        spectrum_path = save_spectrum_as_hdf5(
            id=data.id,
            spectrum=data.spectrum,
        )

        save_inference_result(
            id=data.id,
            k_estimate=data.k_estimate,
            doa=data.doa,
            raw_format=data.raw_format,
            input_timestamp=data.input_timestamp,
            output_timestamp=data.output_timestamp,
            spectrum_path=spectrum_path,
            snr_estimate=data.snr_estimate,
            confidence=data.confidence,
        )

        return {
            "id": data.id,
            "raw_format": data.raw_format,
            "input_timestamp": data.input_timestamp.isoformat(),
            "output_timestamp": data.output_timestamp.isoformat(),
            "k_estimate": data.k_estimate,
            "doa": data.doa,
            "spectrum_path": spectrum_path,
            "snr_estimate": data.snr_estimate,
            "confidence": data.confidence,
            "message": "Inference result saved",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#최근 raw signal 조회
@app.get("/raw-signals/recent")
def read_recent_raw_signals(limit: int = 20):
    return get_raw_signals(limit=limit)

#최근 추론 결과 조회
@app.get("/inference-results/latest")
def read_latest_inference_result():
    result = get_latest_inference_result()

    if result is None:
        raise HTTPException(status_code=404, detail="No inference result found")

    return result

#여러 결과 조회
@app.get("/inference-results/recent")
def read_recent_inference_results(limit: int = 20):
    return get_recent_inference_results(limit=limit)

#시간 기준 조회
@app.get("/inference-results/time-range")
def read_results_by_time_range(start: str = "-24h", stop: Optional[str] = None):
    return get_results_by_time_range(start=start, stop=stop)

#id로 조회
@app.get("/signals/{id}/results")
def read_results_by_id(id: str):
    return get_results_by_id(id=id)

#통계 보기
@app.get("/history/summary")
def read_history_summary(hours: int = 24):
    return get_history_summary(hours=hours)