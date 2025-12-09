import os
import time
import cv2
import numpy as np

def save_video_frame(frame_bytes, video_writer, video_size):
    """
    바이트 형태의 프레임을 VideoWriter에 기록

    Parameters
    ----------
    frame_bytes : bytes
        JPEG 바이트 데이터
    video_writer : cv2.VideoWriter
        현재 사용 중인 VideoWriter 객체
    video_size : tuple
        (width, height)
    """
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if video_writer is not None and img is not None:
        video_writer.write(img)

def get_new_video_writer(video_dir, video_idx, video_size, fps=30):
    """
    새로운 VideoWriter 생성

    Parameters
    ----------
    video_dir : str
        영상 저장 폴더
    video_idx : int
        영상 파일 번호
    video_size : tuple
        (width, height)
    fps : int
        저장 영상 FPS
    """
    os.makedirs(video_dir, exist_ok=True)
    filename = f"video_{video_idx:04d}.mp4"
    save_path = os.path.join(video_dir, filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, video_size)
    return writer
