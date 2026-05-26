# server/model_loader.py -> 서버 시작 시, 모델을 한 번만 로드

import os
import numpy as np
import torch

from model import LCPSN
from physics import ULA_action_vector
from criterion import build_angle_grid_rad


M = 8
SNAPSHOTS = 200
ANGLES_COUNT = 360
K_MIN = 2
K_MAX = 5
VARIANCE_FLOOR = 5e-2

CHECKPOINT_PATH = "best_lcpsn_1bit.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
a_steering = None
angle_grid_rad = None


def precompute_steering_vectors(m: int, angles_count: int = 360):
    angles = np.linspace(-np.pi / 2, np.pi / 2, angles_count, endpoint=False)
    a_matrix = torch.zeros((m, angles_count), dtype=torch.complex64)

    for i, ang in enumerate(angles):
        a_matrix[:, i] = torch.from_numpy(
            ULA_action_vector(ang, m)
        ).to(torch.complex64)

    return a_matrix.to(DEVICE)


def load_model_once():
    global model, a_steering, angle_grid_rad

    if model is not None:
        return

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    model = LCPSN(
        m=M,
        snapshots=SNAPSHOTS,
        angles_count=ANGLES_COUNT,
        d_model=96,
        nhead=8,
        num_layers=4,
        ff_dim=256,
        dropout=0.10,
        spectrum_hidden=512,
        k_min=K_MIN,
        k_max=K_MAX,
        variance_floor=VARIANCE_FLOOR,
    ).to(DEVICE)

    state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    a_steering = precompute_steering_vectors(M, ANGLES_COUNT)
    angle_grid_rad = build_angle_grid_rad(ANGLES_COUNT, device=DEVICE)

    print(f"[MODEL] Loaded checkpoint: {CHECKPOINT_PATH}")
    print(f"[MODEL] Device: {DEVICE}")