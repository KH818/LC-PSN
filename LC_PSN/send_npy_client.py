import time
import requests
import numpy as np
import tempfile
import os

from physics import construct_signal
from dataset import sample_thetas_with_min_sep

URL = "http://127.0.0.1:8000/infer-npy"

M = 8
SNAPSHOTS = 200
SNR = 5 # SNR은 고정

K_MIN = 2
K_MAX = 5
MIN_SEP_DEG = 8.0


while True:
    # 1. 신호원 개수 K를 랜덤 선택
    k_true = np.random.randint(K_MIN, K_MAX + 1)

    # 2. 최소 간격을 만족하는 DOA 각도 랜덤 생성
    # 반환값은 radian 단위
    thetas_rad = sample_thetas_with_min_sep(
        num=k_true,
        min_sep_deg=MIN_SEP_DEG
    )

    # 확인용 degree 변환
    true_doas_deg = np.rad2deg(thetas_rad).round(2).tolist()

    # 3. 실제 DOA 구조를 가진 수신 행렬 생성
    # X shape: [8, 200], complex
    X, _ = construct_signal(
        thetas=thetas_rad,
        snr=SNR,
        snapshots=SNAPSHOTS,
        m=M
    )

    X = X.astype(np.complex64)

    # 4. .npy 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
        np.save(tmp.name, X)
        tmp_path = tmp.name

    try:
        # 5. FastAPI 서버로 POST 요청
        with open(tmp_path, "rb") as f:
            files = {
                "file": ("sample_input.npy", f, "application/octet-stream")
            }

            data = {
                "sensor_id": "ula_01"
            }

            response = requests.post(URL, files=files, data=data)

        result = response.json()

        print("true_k:", k_true)
        print("true_doa_deg:", true_doas_deg)

        # 서버가 졸업작품.md 형식으로 event를 반환하는 경우
        if "output" in result:
            print("pred_k:", result["output"].get("k_estimate"))
            print("pred_doa_deg:", result["output"].get("doa_deg"))
            print("k_confidence:", result["output"].get("k_confidence"))
        else:
            # 아직 Mock 결과 형식이면 여기로 출력됨
            print("server_response:", result)

        print("-" * 80)

    finally:
        os.remove(tmp_path)

    time.sleep(3)

    # -- 시나리오 예시 ---
    #  K 랜덤 선택, 예: 3
    # → DOA 랜덤 선택, 예: [-62.3, 4.8, 41.2]
    # → construct_signal()로 [8, 200] complex 수신 행렬 생성
    # → .npy 파일로 서버에 전송
    # → 서버가 [1, 16, 200]으로 전처리
    # → 모델 추론
    # → output.doa_deg와 true_doa_deg 비교