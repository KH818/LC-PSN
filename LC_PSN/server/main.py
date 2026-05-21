# server/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from server.schemas import PredictRequest, PredictResponse
from server.mock_inference import mock_predict
from server.websocket_manager import WebSocketManager
# from server.file_preprocess import load_first_sample_from_h5
from server.file_preprocess import npy_complex_to_model_input
# from server.stream_inference import H5StreamInference
import os
import shutil
import tempfile
import h5py
import numpy as np
import asyncio

app = FastAPI(
    title="DOA Estimation Server",
    description="FastAPI server for LC-PSN DOA estimation",
    version="0.1.0"
)

ws_manager = WebSocketManager()

@app.post("/infer-npy")
async def infer_npy(file: UploadFile = File(...)):
    if not file.filename.endswith(".npy"):
        raise HTTPException(status_code=400, detail="Only .npy files are supported")

    try:
        file_bytes = await file.read()
        x = npy_complex_to_model_input(file_bytes, use_1bit=True)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = mock_predict()
    result["input_file"] = file.filename
    result["input_shape"] = list(x.shape)
    result["input_format"] = "complex_npy"

    await ws_manager.broadcast({
        "type": "inference_result",
        "data": result
    })

    return result

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