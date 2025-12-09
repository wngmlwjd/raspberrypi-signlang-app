import os
import numpy as np

from config.config import (
    SEQUENCE_LENGTH, SEQUENCE_STEP,
    LANDMARKS_DIR, FEATURES_DIR, MAXJ_PATH
)
from utils import log_message


# ----------------------------------------------------
# 1. 좌표 정규화 함수 (단일 프레임 또는 복수 프레임 모두 지원)
# ----------------------------------------------------
def transform_and_normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    landmarks : (J, 3) 또는 (T, J, 3)
    """

    arr = np.array(landmarks)

    if arr.ndim == 2:        # (J, 3)
        arr = arr[np.newaxis, :, :]

    if arr.size == 0:
        return arr

    # 기준점 (첫 번째 관절)
    ref = arr[:, 0:1, :]         # (T,1,3)

    # 기준 이동
    transformed = arr - ref

    # 크기 정규화
    max_norm = np.max(np.linalg.norm(transformed, axis=-1, keepdims=True))

    if max_norm < 1e-6:
        return transformed

    return transformed / max_norm


# ----------------------------------------------------
# 2. 실시간 단일 프레임 전처리 함수
# ----------------------------------------------------
def process_to_feature(landmarks: np.ndarray) -> np.ndarray:
    """
    실시간 추론에서 사용
    landmarks : (J, 3)
    return    : (J_max * 3,) flatten된 feature
    """

    if landmarks is None:
        return None

    # Load J_max
    if not os.path.exists(MAXJ_PATH):
        raise FileNotFoundError(f"maxJ file not found: {MAXJ_PATH}")

    try:
        if MAXJ_PATH.endswith(".txt"):
            with open(MAXJ_PATH, "r") as f:
                J_max = int(f.read().strip())
        else:
            J_max = int(np.load(MAXJ_PATH))
    except:
        raise RuntimeError("Failed to load maxJ")

    arr = np.array(landmarks)     # (J, 3)

    # pad or cut
    J = arr.shape[0]
    if J < J_max:
        arr = np.pad(arr, ((0, J_max - J), (0, 0)), mode="constant")
    elif J > J_max:
        arr = arr[:J_max, :]

    # normalize
    arr = transform_and_normalize_landmarks(arr)[0]  # (J_max, 3)

    # flatten → 모델 입력용 shape = (T, J_max*3)
    return arr.reshape(-1)


# ----------------------------------------------------
# 3. 단일 영상 → 여러 feature 생성 (슬라이딩 윈도우)
# ----------------------------------------------------
def generate_features_with_sliding(
    frame_npy_dir: str = LANDMARKS_DIR,
    save_dir: str = FEATURES_DIR,
    maxj_path: str = MAXJ_PATH
) -> int:
    """
    frame_npy_dir : 단일 영상에서 추출된 frame landmark npy들이 들어있는 폴더
    save_dir      : 슬라이딩 윈도우로 자른 feature 저장 폴더
    """

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

    # -----------------------------
    # Normalize all frames (멀티프레임 npy도 지원)
    # -----------------------------
    normalized_frames = []

    for fname in npy_files:
        arr = np.load(os.path.join(frame_npy_dir, fname))   # (J,3) or (T,J,3)

        if arr.size == 0:
            continue

        # 2차원 → (1,J,3)로 변환
        if arr.ndim == 2:  
            arr = arr[np.newaxis, :, :]  # (1,J,3)

        # 3차원, 첫 번째 차원이 1 이상이면 그대로 사용
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Invalid landmark shape: {arr.shape}")

        # pad or truncate J
        J = arr.shape[1]
        if J < J_max:
            arr = np.pad(arr, ((0, 0), (0, J_max - J), (0, 0)), mode="constant")
        elif J > J_max:
            arr = arr[:, :J_max, :]

        # normalize
        arr = transform_and_normalize_landmarks(arr)  # (T,J_max,3)
        
        # 모든 프레임을 리스트에 추가
        for f in arr:
            normalized_frames.append(f)

    total_frames = len(normalized_frames)

    # -----------------------------
    # Sliding window (기존과 동일)
    # -----------------------------
    os.makedirs(save_dir, exist_ok=True)
    feature_count = 0

    start = 0
    while start < total_frames:
        end = start + SEQUENCE_LENGTH
        seq = normalized_frames[start:end]

        # 부족하면 제로패딩
        if len(seq) < SEQUENCE_LENGTH:
            pad_len = SEQUENCE_LENGTH - len(seq)
            seq += [np.zeros((J_max, 3), dtype=np.float32) for _ in range(pad_len)]

        seq_array = np.stack(seq, axis=0)          # (T,J_max,3)
        seq_flat = seq_array.reshape(SEQUENCE_LENGTH, -1)

        save_path = os.path.join(save_dir, f"seq_{feature_count:04d}.npy")
        np.save(save_path, seq_flat)

        log_message(f"Saved seq: {save_path}")

        feature_count += 1
        start += SEQUENCE_STEP

    log_message(f"Total sequences saved: {feature_count}")
    return feature_count
