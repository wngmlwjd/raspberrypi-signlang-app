import os
import numpy as np

from config.config import (
    LANDMARKS_DIR, FEATURES_DIR, MAXJ_PATH, DATA_SIZE
)
from utils import log_message


# ----------------------------------------------------
# 좌표 정규화 (기존 동일)
# ----------------------------------------------------
def transform_and_normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    arr = np.array(landmarks)

    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]

    if arr.size == 0:
        return arr

    ref = arr[:, 0:1, :]
    transformed = arr - ref

    max_norm = np.max(np.linalg.norm(transformed, axis=-1, keepdims=True))
    if max_norm < 1e-6:
        return transformed

    return transformed / max_norm


# ----------------------------------------------------
# 단일 파일 전처리 (DATA_SIZE 패딩 추가)
# ----------------------------------------------------
def generate_single_feature(
    frame_npy_dir: str = LANDMARKS_DIR,
    save_path: str = None,
    maxj_path: str = MAXJ_PATH
) -> str:

    if save_path is None:
        os.makedirs(FEATURES_DIR, exist_ok=True)
        save_path = os.path.join(FEATURES_DIR, "feature_single.npy")

    # -----------------------------
    # Load maxJ
    # -----------------------------
    if not os.path.exists(maxj_path):
        raise FileNotFoundError(f"maxJ file not found: {maxj_path}")

    try:
        if maxj_path.endswith(".txt"):
            with open(maxj_path, "r") as f:
                J_max = int(f.read().strip())
        else:
            J_max = int(np.load(maxj_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load maxJ: {e}")

    log_message(f"Loaded maxJ: {J_max}")

    # -----------------------------
    # Load npy frame list
    # -----------------------------
    npy_files = sorted([f for f in os.listdir(frame_npy_dir) if f.endswith(".npy")])
    if not npy_files:
        raise FileNotFoundError(f"No .npy frames found: {frame_npy_dir}")

    log_message(f"Found {len(npy_files)} frame npy files")

    processed_frames = []

    # -----------------------------
    # Normalize all frames
    # -----------------------------
    for fname in npy_files:
        arr = np.load(os.path.join(frame_npy_dir, fname))

        if arr.size == 0:
            continue

        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]
        elif arr.ndim != 3:
            raise ValueError(f"Invalid landmark shape: {arr.shape}")

        J = arr.shape[1]
        if J < J_max:
            arr = np.pad(arr, ((0, 0), (0, J_max - J), (0, 0)), mode="constant")
        elif J > J_max:
            arr = arr[:, :J_max, :]

        arr = transform_and_normalize_landmarks(arr)

        for f in arr:
            processed_frames.append(f)

    total_frames = len(processed_frames)
    log_message(f"Total normalized frames: {total_frames}")

    # --------------------------------------------------------------
    # DATA_SIZE 만큼 패딩
    # --------------------------------------------------------------
    if total_frames < DATA_SIZE:
        pad_len = DATA_SIZE - total_frames

        log_message(f"Padding applied: {pad_len} frames")

        pad_frames = [np.zeros((J_max, 3), dtype=np.float32) for _ in range(pad_len)]
        processed_frames.extend(pad_frames)

    # --------------------------------------------------------------
    # shape = (DATA_SIZE or F, J_max*3)
    # --------------------------------------------------------------
    output_frames = len(processed_frames)

    all_array = np.array(processed_frames)           # (N, J_max, 3)
    all_flat = all_array.reshape(output_frames, -1)  # (N, J_max*3)

    np.save(save_path, all_flat)
    log_message(f"Saved single feature: {save_path}")

    return save_path
