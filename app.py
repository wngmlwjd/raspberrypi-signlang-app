import time
from flask import Flask
from camera.camera_stream import CameraStream
from config.config import CMD, FRAMES_DIR
from inference.video_saver import save_frame
import os

app = Flask(__name__)

# 프리뷰 없이 카메라 실행하도록 CMD 수정
if "--nopreview" not in CMD:
    CMD.append("--nopreview")

# 카메라 스트림 시작
camera = CameraStream(cmd=CMD)

frame_idx = 0
os.makedirs(FRAMES_DIR, exist_ok=True)

@app.route("/")
def index():
    return "카메라 프레임 저장 서버 실행 중"

def capture_loop():
    global frame_idx
    while True:
        frame = camera.get_frame()
        if frame is not None:
            save_frame(frame, FRAMES_DIR, frame_idx)
            frame_idx += 1
        time.sleep(0.01)  # CPU 점유율 줄이기

if __name__ == "__main__":
    try:
        # 프레임 저장 루프를 백그라운드에서 실행
        import threading
        threading.Thread(target=capture_loop, daemon=True).start()
        
        # Flask 서버 실행
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        camera.stop()
        print(f"총 {frame_idx}개의 프레임을 저장했습니다 → {FRAMES_DIR}")
