# server/main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from server.schemas import PredictRequest, PredictResponse
from server.mock_inference import mock_predict
from server.websocket_manager import WebSocketManager

app = FastAPI(
    title="DOA Estimation Inference Server",
    description="FastAPI server for LC-PSN / TransMUSIC DOA estimation",
    version="0.1.0"
)

ws_manager = WebSocketManager()


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


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    result = mock_predict()

    await ws_manager.broadcast({
        "type": "inference_result",
        "data": result
    })

    return result


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