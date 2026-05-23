import math
import random
from typing import Dict, List, Optional


ANGLE_GRID = list(range(-90, 91))
TARGET_DOA = 42.0
_fallback_frame_index = 0


def _jitter(value: float, amount: float = 1.2) -> float:
    return value + random.uniform(-amount, amount)


def _gaussian_peak(angle: float, center: float, width: float, amplitude: float) -> float:
    distance = angle - center
    return amplitude * math.exp(-(distance**2) / (2 * width**2))


def generate_mock_spectrum(sources: List[Dict], grid_size: int = 181) -> List[float]:
    """
    -90도부터 +90도까지의 angle grid에 대해 scenario source peak를 만든다.
    grid_size는 호환성을 위해 남겨두지만 현재는 1도 간격 181개 grid를 사용한다.
    """
    del grid_size
    spectrum = []

    for angle in ANGLE_GRID:
        value = 0.018 + random.random() * 0.018

        for source in sources:
            value += _gaussian_peak(
                angle=angle,
                center=source["doa"],
                width=source.get("width", 3.5),
                amplitude=source["score"],
            )

        spectrum.append(round(min(value, 1.0), 4))

    return spectrum


def _build_sources(frame_index: int) -> List[Dict]:
    phase_frame = frame_index % 48
    jammer_active = phase_frame >= 28
    new_source_active = phase_frame >= 12
    decoy_source_active = 20 <= phase_frame < 36

    target_score = 0.92

    if jammer_active:
        # 재밍 효과를 보여주기 위해 target 방향 peak가 점진적으로 낮아지는 mock 시나리오
        suppression_progress = min(1.0, (phase_frame - 28) / 12)
        target_score = 0.92 - 0.58 * suppression_progress

    sources = [
        {
            "id": "track_a",
            "doa": _jitter(-31.5),
            "score": 0.82 + random.uniform(-0.04, 0.04),
            "width": 3.8,
        },
        {
            "id": "target_b",
            "doa": _jitter(TARGET_DOA),
            "score": target_score + random.uniform(-0.035, 0.035),
            "width": 3.2,
        },
    ]

    if new_source_active:
        sources.append(
            {
                "id": "track_c",
                "doa": _jitter(9.5),
                "score": 0.58 + random.uniform(-0.05, 0.05),
                "width": 4.4,
            }
        )

    if decoy_source_active:
        sources.append(
            {
                "id": "intermittent_d",
                "doa": _jitter(-8.0, amount=2.2),
                "score": 0.42 + random.uniform(-0.06, 0.06),
                "width": 5.0,
            }
        )

    return sorted(sources, key=lambda source: source["doa"])


def mock_predict(frame_index: Optional[int] = None) -> Dict:
    """
    실제 LC-PSN 모델 대신 사용하는 scenario 기반 mock 추론 함수.
    반환 데이터 규격은 기존 mock API와 동일하게 유지한다.
    """
    global _fallback_frame_index

    if frame_index is None:
        frame_index = _fallback_frame_index
        _fallback_frame_index += 1

    phase_frame = frame_index % 48
    jammer_active = phase_frame >= 28
    sources = _build_sources(frame_index)
    doas_deg = [round(source["doa"], 1) for source in sources]
    spectrum = generate_mock_spectrum(sources)
    confidence_base = 0.9 if not jammer_active else 0.84
    confidence = confidence_base + random.uniform(-0.035, 0.025)

    return {
        "estimated_k": len(doas_deg),
        "doas_deg": doas_deg,
        "confidence": round(max(0.55, min(confidence, 0.98)), 3),
        "spectrum": spectrum,
        "message": "Mock inference success",
    }
