import subprocess
import cv2
import numpy as np

cmd = [
    "raspi-vid", "-t", "0", "-o", "test.h264", "-w", "640", "-h", "480", "-fps", "30"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

# 스트림 처리 (예시: mjpeg 또는 H.264 디코딩 필요)
