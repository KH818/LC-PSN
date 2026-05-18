# server/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from server.schemas import PredictRequest, PredictResponse
from server.mock_inference import mock_predict
from server.websocket_manager import WebSocketManager
from server.file_preprocess import load_first_sample_from_h5
from server.stream_inference import H5StreamInference
import os
import shutil
import tempfile
import h5py
import numpy as np
import asyncio

ws_manager = WebSocketManager()

STREAM_H5_PATH = "train_mini_1bit.h5"
streamer = H5StreamInference(STREAM_H5_PATH, interval_sec=3.0)
stream_task = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global stream_task

    print("[SERVER] Starting background stream inference...")
    stream_task = asyncio.create_task(streamer.run(ws_manager))

    yield

    print("[SERVER] Stopping background stream inference...")
    streamer.stop()

    if stream_task:
        stream_task.cancel()


app = FastAPI(
    title="DOA Estimation Inference Server",
    description="FastAPI server for LC-PSN / TransMUSIC DOA estimation",
    version="0.1.0",
    lifespan=lifespan
)

@app.post("/infer-file")
async def infer_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="Only .h5 files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        x = load_first_sample_from_h5(tmp_path, use_1bit=True)

        result = mock_predict() # 실제 모델 추론 대신 Mock 결과 사용
        result["input_file"] = file.filename
        result["input_shape"] = list(x.shape)

        await ws_manager.broadcast({
            "type": "inference_result",
            "data": result
        })

        return result

    finally:
        os.remove(tmp_path)

@app.post("/inspect-h5")
async def inspect_h5(file: UploadFile = File(...)):
    if not file.filename.endswith(".h5"):
        raise HTTPException(status_code=400, detail="Only .h5 files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with h5py.File(tmp_path, "r") as hf:
            keys = list(hf.keys())

            result = {
                "filename": file.filename,
                "keys": keys
            }

            if "X" in hf:
                X = hf["X"]
                result["X_shape"] = list(X.shape)
                result["X_dtype"] = str(X.dtype)

            if "Y" in hf:
                Y = hf["Y"]
                result["Y_shape"] = list(Y.shape)
                result["Y_dtype"] = str(Y.dtype)

        return result

    finally:
        os.remove(tmp_path)

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