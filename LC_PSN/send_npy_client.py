import json
import os
import tempfile
import time
import urllib.request
import uuid

import numpy as np

from physics import construct_signal


URL = "http://127.0.0.1:8000/infer-npy"

M = 8
SNAPSHOTS = 200
K_MIN = 2
K_MAX = 5
MIN_SEP_DEG = 8.0
SEND_INTERVAL_SEC = 3


SCENARIO_TRACKS = [
    {
        "name": "northwest_patrol",
        "start_frame": 0,
        "end_frame": None,
        "start_deg": -42.0,
        "velocity_deg": 0.35,
        "jitter_deg": 0.55,
    },
    {
        "name": "east_fixed",
        "start_frame": 0,
        "end_frame": None,
        "start_deg": 31.0,
        "velocity_deg": -0.12,
        "jitter_deg": 0.45,
    },
    {
        "name": "new_contact",
        "start_frame": 8,
        "end_frame": 34,
        "start_deg": 8.0,
        "velocity_deg": 0.28,
        "jitter_deg": 0.7,
    },
    {
        "name": "short_emitter",
        "start_frame": 20,
        "end_frame": 46,
        "start_deg": 57.0,
        "velocity_deg": -0.18,
        "jitter_deg": 0.65,
    },
]


def clamp_angle(deg):
    return float(np.clip(deg, -85.0, 85.0))


def is_track_active(track, frame_index):
    if frame_index < track["start_frame"]:
        return False

    return track["end_frame"] is None or frame_index <= track["end_frame"]


def keep_min_separation(doas_deg, min_sep_deg):
    selected = []

    for deg in sorted(doas_deg):
        if all(abs(deg - existing) >= min_sep_deg for existing in selected):
            selected.append(deg)

    return selected[:K_MAX]


def get_scenario_doas(frame_index):
    doas_deg = []

    for track in SCENARIO_TRACKS:
        if not is_track_active(track, frame_index):
            continue

        age = frame_index - track["start_frame"]
        drift = track["velocity_deg"] * age
        slow_wave = 1.2 * np.sin((frame_index + len(track["name"])) / 7.0)
        jitter = np.random.normal(0.0, track["jitter_deg"])
        doas_deg.append(clamp_angle(track["start_deg"] + drift + slow_wave + jitter))

    doas_deg = keep_min_separation(doas_deg, MIN_SEP_DEG)

    if len(doas_deg) < K_MIN:
        fallback = [-45.0, 30.0]
        doas_deg = keep_min_separation(doas_deg + fallback, MIN_SEP_DEG)

    return doas_deg


def get_scenario_snr(frame_index):
    base_snr = 5.0
    slow_fading = 1.5 * np.sin(frame_index / 9.0)
    small_noise = np.random.normal(0.0, 0.35)

    return float(np.clip(base_snr + slow_fading + small_noise, 1.0, 10.0))


def save_temp_npy(x):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp:
        np.save(tmp.name, x)
        return tmp.name


def build_multipart_body(file_path, sensor_id):
    boundary = f"----lcpsn{uuid.uuid4().hex}"
    line_break = "\r\n"

    with open(file_path, "rb") as file:
        file_bytes = file.read()

    body = b"".join(
        [
            f"--{boundary}{line_break}".encode(),
            b'Content-Disposition: form-data; name="sensor_id"',
            f"{line_break}{line_break}{sensor_id}{line_break}".encode(),
            f"--{boundary}{line_break}".encode(),
            b'Content-Disposition: form-data; name="file"; filename="sample_input.npy"',
            f"{line_break}Content-Type: application/octet-stream{line_break}{line_break}".encode(),
            file_bytes,
            f"{line_break}--{boundary}--{line_break}".encode(),
        ]
    )

    return body, boundary


def post_npy(file_path, sensor_id="ula_01"):
    body, boundary = build_multipart_body(file_path, sensor_id)
    request = urllib.request.Request(
        URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def print_result(frame_index, true_doas_deg, snr, result):
    print(f"frame: {frame_index}")
    print("true_k:", len(true_doas_deg))
    print("true_doa_deg:", [round(deg, 2) for deg in true_doas_deg])
    print("scenario_snr:", round(snr, 2))

    if "output" in result:
        print("event_id:", result.get("event_id"))
        print("pred_k:", result["output"].get("k_estimate"))
        print("pred_doa_deg:", result["output"].get("doa_deg"))
        print("k_confidence:", result["output"].get("k_confidence"))
    else:
        print("server_response:", result)

    print("-" * 80)


def main():
    frame_index = 0

    while True:
        true_doas_deg = get_scenario_doas(frame_index)
        thetas_rad = np.deg2rad(true_doas_deg)
        snr = get_scenario_snr(frame_index)

        x, _ = construct_signal(
            thetas=thetas_rad,
            snr=snr,
            snapshots=SNAPSHOTS,
            m=M,
        )
        x = x.astype(np.complex64)
        tmp_path = save_temp_npy(x)

        try:
            result = post_npy(tmp_path)
            print_result(frame_index, true_doas_deg, snr, result)
        finally:
            os.remove(tmp_path)

        frame_index += 1
        time.sleep(SEND_INTERVAL_SEC)


if __name__ == "__main__":
    main()
