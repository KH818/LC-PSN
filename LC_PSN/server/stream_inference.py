# server/stream_inference.py

import asyncio
import h5py
import numpy as np
import torch
from typing import Optional

from server.mock_inference import mock_predict


class H5StreamInference:
    def __init__(self, h5_path: str, interval_sec: float = 3.0): # 3초마다 샘플 하나씩 처리
        self.h5_path = h5_path
        self.interval_sec = interval_sec
        self.X = None
        self.index = 0
        self.total = 0
        self.running = False

    def load_file(self):
        """
        h5 파일을 미리 로드한다.
        X shape: [N, 8, 200] complex64
        """
        with h5py.File(self.h5_path, "r") as hf:
            if "X" not in hf:
                raise ValueError("h5 file must contain dataset 'X'")

            self.X = np.array(hf["X"])

        self.total = self.X.shape[0]
        self.index = 0

        print(f"[STREAM] Loaded {self.h5_path}")
        print(f"[STREAM] X shape = {self.X.shape}")

    def get_sample_tensor(self, idx: int) -> torch.Tensor:
        """
        X[idx] 하나를 LC-PSN 입력 형태 [1, 16, 200] tensor로 변환한다.
        """
        if self.X is None:
            raise RuntimeError("H5 file is not loaded")

        X_raw = self.X[idx:idx + 1]  # [1, 8, 200] complex

        X_processed = np.concatenate(
            [X_raw.real, X_raw.imag],
            axis=1
        ).astype(np.float32)  # [1, 16, 200]

        X_processed = np.sign(X_processed).astype(np.float32)
        X_processed[X_processed == 0] = 1.0

        return torch.from_numpy(X_processed).float()

    async def run(self, websocket_manager):
        """
        몇 초마다 자동으로 샘플 하나씩 처리하고 WebSocket으로 전송한다.
        """
        if self.X is None:
            self.load_file()

        self.running = True

        while self.running:
            x = self.get_sample_tensor(self.index)

            # 지금은 실제 모델 대신 Mock 결과 사용
            result = mock_predict()
            result["input_file"] = self.h5_path
            result["sample_index"] = self.index
            result["input_shape"] = list(x.shape)
            result["streaming"] = True

            await websocket_manager.broadcast({
                "type": "stream_inference_result",
                "data": result
            })

            print(f"[STREAM] Sent sample index {self.index}")

            self.index += 1

            if self.index >= self.total:
                self.index = 0

            await asyncio.sleep(self.interval_sec)

    def stop(self):
        self.running = False