import subprocess
import os
import time

from config.config import RAW_DIR

counter = 0

while True:
    # 파일 이름 생성 (test_000.mp4, test_001.mp4, ...)
    video_filename = os.path.join(RAW_DIR, f"test_{counter:03d}.mp4")
    
    # 5초 동안 비디오 녹화
    subprocess.run(["rpicam-vid", "-t", "5000", "-o", video_filename])
    
    counter += 1
    # 녹화가 끝나면 바로 다음 영상 녹화
    # 필요하면 sleep으로 간격 조절 가능
