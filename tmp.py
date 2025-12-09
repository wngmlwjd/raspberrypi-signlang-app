import subprocess
import os

from config.config import FRAMES_DIR

counter = 0

while counter < 10:
    filename = f"test_{counter:03d}.jpg"
    # timeout을 1ms로 설정 → 사실상 바로 촬영 후 저장
    subprocess.run(["rpicam-jpeg", "--output", os.path.join(FRAMES_DIR, filename), "--timeout", "1"])
    counter += 1
