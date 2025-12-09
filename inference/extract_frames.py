import cv2
import os
from utils import log_message
from config import config

def extract_frames(video_path=None, output_dir="frames"):
    """
    영상에서 모든 프레임 추출 후 output_dir에 저장
    """
    if video_path is None:
        video_path = config.VIDEO_PATH

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    log_message(f"[Frames] {video_path} -> {frame_count} frames saved to {output_dir}")
    return frame_count
