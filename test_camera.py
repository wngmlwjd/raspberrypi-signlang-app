import subprocess
import cv2
import numpy as np

cmd = [
    "rpicam-vid", "-t", "0", "-o", "test.h264", "--width", "640", "--height", "480", "--framerate", "30"
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

# 스트림 처리 (예시: mjpeg 또는 H.264 디코딩 필요)
