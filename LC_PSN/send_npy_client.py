import time
import requests
import numpy as np
import tempfile
import os

URL = "http://127.0.0.1:8000/infer-npy"

while True:
    # 예시용 복소수 수신 행렬 [8, 200]
    X = np.random.randn(8, 200) + 1j * np.random.randn(8, 200)
    X = X.astype(np.complex64)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
        np.save(tmp.name, X)
        tmp_path = tmp.name

    try: # 임시로 만든 .npy 파일을 서버(/infer-npy)에 POST 요청
        with open(tmp_path, "rb") as f:
            files = {
                "file": ("sample_input.npy", f, "application/octet-stream")
            }
            response = requests.post(URL, files=files)

        print(response.json())

    finally:
        os.remove(tmp_path)

    time.sleep(3)