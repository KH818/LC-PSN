import random
import math
from typing import Dict, List


def generate_mock_spectrum(doas_deg: List[float], grid_size: int = 181) -> List[float]:
    """
    -90도부터 +90도까지 1도 간격으로 가짜 spectrum 생성
    DOA 근처에서 peak가 생기도록 만듦
    """
    angle_grid = list(range(-90, 91))
    spectrum = []

    for angle in angle_grid:
        value = 0.02 + random.random() * 0.03

        for doa in doas_deg:
            distance = angle - doa
            value += math.exp(-(distance ** 2) / (2 * 3.0 ** 2))

        spectrum.append(round(min(value, 1.0), 4))

    return spectrum


def mock_predict() -> Dict:
    """
    실제 LC-PSN 모델 대신 사용할 Mock 추론 함수
    """
    estimated_k = random.randint(2, 5)

    doas_deg = sorted(
        random.sample(range(-70, 71), estimated_k)
    )

    spectrum = generate_mock_spectrum(doas_deg)

    return {
        "estimated_k": estimated_k,
        "doas_deg": doas_deg,
        "confidence": round(random.uniform(0.75, 0.98), 3),
        "spectrum": spectrum,
        "message": "Mock inference success"
    }