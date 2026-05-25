# server/main.py

from datetime import datetime, timezone
from contextlib import asynccontextmanager
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException

from server.websocket_manager import WebSocketManager
from server.file_preprocess import npy_complex_to_model_input
from server.model_loader import load_model_once
from server.real_inference import real_model_predict


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
        x = npy_complex_to_model_input(file_bytes, use_1bit=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raw_data_id = f"raw_{uuid.uuid4().hex[:12]}"

    event = real_model_predict(
        x,
        raw_data_id=raw_data_id,
        sensor_id=sensor_id,
        server_received_at=server_received_at,
    )

    await ws_manager.broadcast({
        "type": "doa_inference_event",
        "data": event
    })

    return event


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