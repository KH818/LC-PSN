# server/api_adapter.py
# 내부 Output을 Event 형태(JSON)로 변환
# 졸업작품.md의 9.2/9.3 참고

import math
import uuid
from datetime import datetime, timezone

import torch


def select_topk_separated_indices(
    score: torch.Tensor,
    angle_grid_rad: torch.Tensor,
    k_hat: int,
    min_separation_deg: float = 8.0,
):
    score_cpu = score.detach().cpu()
    grid_cpu = angle_grid_rad.detach().cpu()
    min_sep = math.radians(min_separation_deg)

    candidates = []

    for i in range(1, score_cpu.numel() - 1):
        if score_cpu[i] > score_cpu[i - 1] and score_cpu[i] > score_cpu[i + 1]:
            candidates.append(i)

    if not candidates:
        candidates = torch.argsort(score_cpu, descending=True).tolist()

    selected = []

    for idx in sorted(candidates, key=lambda j: float(score_cpu[j]), reverse=True):
        if all(abs(float(grid_cpu[idx] - grid_cpu[j])) >= min_sep for j in selected):
            selected.append(idx)

        if len(selected) >= k_hat:
            break

    if len(selected) < k_hat:
        for idx in torch.argsort(score_cpu, descending=True).tolist():
            if idx in selected:
                continue

            if all(abs(float(grid_cpu[idx] - grid_cpu[j])) >= min_sep for j in selected):
                selected.append(idx)

            if len(selected) >= k_hat:
                break

    return sorted(selected[:k_hat])


def format_lcpsn_response(
    *,
    out,
    x,
    angle_grid_rad,
    raw_data_id,
    sensor_id,
    checkpoint_name,
    server_received_at,
    inference_finished_at,
    k_min=2,
    k_max=5,
    min_separation_deg=8.0,
):
    """
    LC-PSN internal output을 FastAPI/WebSocket용 DOAInferenceEvent dict로 변환한다.
    """
    mu = out["mu"][0].detach().cpu()
    k_logits = out["K_logits"][0].detach().cpu()

    k_prob = torch.softmax(k_logits, dim=-1)
    k_hat = int(torch.argmax(k_prob).item()) + k_min
    k_hat = max(k_min, min(k_max, k_hat))

    selected_idx = select_topk_separated_indices(
        score=mu,
        angle_grid_rad=angle_grid_rad.detach().cpu(),
        k_hat=k_hat,
        min_separation_deg=min_separation_deg,
    )

    doa_rad = [
        float(angle_grid_rad[i].detach().cpu().item())
        for i in selected_idx
    ]

    doa_deg = [
        float(math.degrees(v))
        for v in doa_rad
    ]

    peak_scores = [
        float(mu[i].item())
        for i in selected_idx
    ]

    spectrum_values = [
        float(v)
        for v in mu.tolist()
    ]

    event = {
        "event_id": f"{datetime.now(timezone.utc).isoformat()}_{uuid.uuid4().hex[:8]}",
        "sensor_id": sensor_id,
        "sensor_timestamp": None,
        "server_received_at": server_received_at.isoformat(),
        "inference_finished_at": inference_finished_at.isoformat(),

        "input": {
            "raw_data_id": raw_data_id,
            "format": "real_augmented_onebit",
            "shape": list(x.shape[-2:]),
            "m": int(x.shape[-2] // 2),
            "snapshots": int(x.shape[-1]),
            "dtype": str(x.dtype).replace("torch.", ""),
            "layout": "[Re_1..Re_M, Im_1..Im_M] x T",
            "values": "not_included",
        },

        "model": {
            "name": "LC-PSN",
            "checkpoint": checkpoint_name,
            "angles_count": int(mu.numel()),
            "k_min": k_min,
            "k_max": k_max,
            "min_separation_deg": min_separation_deg,
        },

        "output": {
            "k_estimate": k_hat,
            "k_confidence": float(k_prob.max().item()),
            "doa_deg": doa_deg,
            "doa_rad": doa_rad,
            "peak_scores": peak_scores,
            "spectrum": {
                "grid_unit": "deg",
                "grid_start": -90.0,
                "grid_end": 90.0,
                "grid_size": int(mu.numel()),
                "values": spectrum_values,
            },
        },

        "diagnostics": {
            "latency_ms": (
                inference_finished_at - server_received_at
            ).total_seconds() * 1000.0,
            "spectrum_max": float(mu.max().item()),
            "spectrum_mean": float(mu.mean().item()),
        },
    }

    return event