import subprocess
import os
from config.config import RAW_DIR

def save_video():
    """
    5초 동안 녹화하고 파일로 저장
    """
    video_filename = os.path.join(RAW_DIR, f"video.mp4")
    
    # 녹화 명령 실행 (비동기 스레드에서 호출됨)
    subprocess.run(["rpicam-vid", "-t", "5000", "-o", video_filename])
