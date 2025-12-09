import os
import cv2
import numpy as np

def save_frame(frame_bytes, save_dir, frame_idx):
    """
    바이트 형태의 프레임을 JPEG 파일로 저장

    Parameters
    ----------
    frame_bytes : bytes
        CameraStream.get_frame()에서 받은 JPEG 바이트
    save_dir : str
        저장 폴더 경로
    frame_idx : int
        파일 이름 인덱스
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"frame_{frame_idx:04d}.jpg"
    save_path = os.path.join(save_dir, filename)
    
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        cv2.imwrite(save_path, img)
