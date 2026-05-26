# server/main.py

from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.websocket_manager import WebSocketManager
from server.file_preprocess import npy_complex_to_model_input
from server.model_loader import load_model_once
from server.real_inference import real_model_predict

from server.hdf5_storage import (
    save_raw_npy_as_hdf5,
    save_spectrum_as_hdf5,
)

from server.influx_client import (
    save_raw_metadata,
    save_inference_result_from_event,
    get_latest_inference_result,
    get_recent_inference_results,
    get_results_by_id,
    get_results_by_time_range,
    get_raw_signals,
    get_history_summary,
    get_results_by_doa_range,
    get_results_by_min_k,
)

ws_manager = WebSocketManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[SERVER] Loading LC-PSN model...")
    load_model_once()
    yield
    print("[SERVER] Shutdown")


app = FastAPI(
    title="DOA Estimation Inference Server",
    description="FastAPI server for LC-PSN DOA estimation",
    version="0.1.0",
    lifespan=lifespan,
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "message": "DOA Estimation Server is running",
        "status": "ok"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/infer-npy")
async def infer_npy(
    file: UploadFile = File(...),
    sensor_id: str = Form("ula_01"),
):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")

    server_received_at = datetime.now(timezone.utc)

    try:
        file_bytes = await file.read()

        raw_data_id = f"raw_{uuid.uuid4().hex[:12]}"

        #raw npy 파일을 HDF5로 저장
        raw_file_path, raw_format, antenna_count, snapshot_count = save_raw_npy_as_hdf5(
            raw_data_id=raw_data_id,
            file_bytes=file_bytes,
            sensor_id=sensor_id,
            received_at=server_received_at,
        )

        #raw signal metadata를 InfluxDB에 저장
        save_raw_metadata(
            raw_data_id=raw_data_id,
            file_path=raw_file_path,
            raw_format=raw_format,
            sensor_id=sensor_id,
            input_timestamp=server_received_at,
            antenna_count=antenna_count,
            snapshot_count=snapshot_count,
            description="Auto saved from POST /infer-npy",
        )

        #모델 입력 형태로 전처리
        x = npy_complex_to_model_input(file_bytes, use_1bit=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    event = real_model_predict(
        x,
        raw_data_id=raw_data_id,
        sensor_id=sensor_id,
        server_received_at=server_received_at,
    )

    #spectrum을 HDF5로 저장
    spectrum_path = save_spectrum_as_hdf5(
        raw_data_id=raw_data_id,
        spectrum=event["output"]["spectrum"]["values"],
    )

    #모델 추론 결과를 InfluxDB에 저장
    save_inference_result_from_event(
        event=event,
        spectrum_path=spectrum_path,
        raw_format=raw_format,
    )

    await ws_manager.broadcast({
        "type": "doa_inference_event",
        "data": event
    })

    return event


#최근 raw signal metadata 조회
@app.get("/raw-signals/recent")
def read_recent_raw_signals(limit: int = 20):
    return get_raw_signals(limit=limit)


#최근 추론 결과 1개 조회
@app.get("/inference-results/latest")
def read_latest_inference_result():
    result = get_latest_inference_result()

    if result is None:
        raise HTTPException(status_code=404, detail="No inference result found")

    return result


#최근 추론 결과 여러 개 조회
@app.get("/inference-results/recent")
def read_recent_inference_results(limit: int = 20):
    return get_recent_inference_results(limit=limit)


#특정 id에 해당하는 추론 결과 조회
@app.get("/signals/{raw_data_id}/results")
def read_results_by_id(raw_data_id: str):
    return get_results_by_id(raw_data_id=raw_data_id)


#시간 범위로 추론 결과 조회
@app.get("/inference-results/time-range")
def read_results_by_time_range(start: str = "-24h", stop: str | None = None):
    return get_results_by_time_range(start=start, stop=stop)


#최근 n 시간에 대한 요약 통계 조회
@app.get("/history/summary")
def read_history_summary(hours: int = 24):
    return get_history_summary(hours=hours)


#DOA 범위로 추론 결과 조회
@app.get("/inference-results/doa-range")
def read_results_by_doa_range(
    min_doa: float,
    max_doa: float,
    limit: int = 100,
):
    return get_results_by_doa_range(
        min_doa=min_doa,
        max_doa=max_doa,
        limit=limit,
    )


#K 개수 기준으로 추론 결과 조회
@app.get("/inference-results/min-k")
def read_results_by_min_k(
    min_k: int,
    limit: int = 100,
):
    return get_results_by_min_k(
        min_k=min_k,
        limit=limit,
    )



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)

    try:
        while True:
            message = await websocket.receive_text()

            if message == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "message": "server alive"
                })

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)

# /infer-npy 동작 흐름
# .npy 파일 수신
# → [8, 200] complex 행렬 읽기
# → [1, 16, 200] tensor 변환
# → LC-PSN 모델 추론
# → DOAInferenceEvent JSON 생성
# → REST 응답 반환
# → WebSocket으로 송출