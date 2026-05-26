# server/real_inference.py -> 실제 모델을 가지고 추론

from datetime import datetime, timezone

import torch

import server.model_loader as ml
from server.api_adapter import format_lcpsn_response


def real_model_predict(
    x: torch.Tensor,
    *,
    raw_data_id: str,
    sensor_id: str,
    server_received_at,
):
    """
    x: [1, 16, 200] tensor
    return: DOAInferenceEvent dict
    """
    if ml.model is None:
        ml.load_model_once()

    x = x.to(ml.DEVICE)

    with torch.no_grad():
        out = ml.model(x, ml.a_steering)

    inference_finished_at = datetime.now(timezone.utc)

    event = format_lcpsn_response(
        out=out,
        x=x,
        angle_grid_rad=ml.angle_grid_rad,
        raw_data_id=raw_data_id,
        sensor_id=sensor_id,
        checkpoint_name=ml.CHECKPOINT_PATH,
        server_received_at=server_received_at,
        inference_finished_at=inference_finished_at,
        k_min=ml.K_MIN,
        k_max=ml.K_MAX,
        min_separation_deg=8.0,
    )

    return event