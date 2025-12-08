import os
import numpy as np

from config.config import FRAMES_DIR, LANDMARKS_DIR
from inference.hand_tracking import HandTracker


# ============================================================
# 1) 실시간 웹캠 프레임에서 랜드마크 추출용 함수
# ============================================================
def extract_landmarks(frame):
    """
    실시간 프레임 입력(frame: numpy array)에서 랜드마크 추출.
    return: (J, 3) 형태의 랜드마크 또는 None
    """
    tracker = HandTracker()
    landmarks = tracker.process_frame(frame)
    tracker.close()

    return landmarks  # (J, 3) or None


# ============================================================
# 2) 기존: 폴더 기반 landmark 생성 기능 (그대로 유지)
# ============================================================
def extract_landmarks_from_frames(frame_dir: str = FRAMES_DIR, save_dir: str = LANDMARKS_DIR):

    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"❌ 프레임 폴더를 찾을 수 없음: {frame_dir}")

    os.makedirs(save_dir, exist_ok=True)
    tracker = HandTracker()

    frame_files = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    if not frame_files:
        raise FileNotFoundError(f"❌ JPG 프레임을 찾을 수 없음: {frame_dir}")

    print(f"📸 총 {len(frame_files)}개의 프레임을 처리합니다...")

    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)

        # 기존 이미지 파일 처리
        landmarks = tracker.process_image(frame_path)

        save_path = os.path.join(
            save_dir,
            os.path.splitext(frame_file)[0] + ".npy"
        )
        tracker.save_landmarks(landmarks, save_path)

    tracker.close()
    print(f"✅ 랜드마크 저장 완료 → {save_dir}")
