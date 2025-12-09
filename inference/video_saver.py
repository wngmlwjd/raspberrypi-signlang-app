import subprocess
import os
import time

from config.config import RAW_DIR

def save_video(counter):
    # 파일 이름 생성 (test_000.mp4, test_001.mp4, ...)
    video_filename = os.path.join(RAW_DIR, f"test_{counter:03d}.mp4")
    
    # 5초 동안 비디오 녹화
    subprocess.run(["rpicam-vid", "-t", "5000", "-o", video_filename])
