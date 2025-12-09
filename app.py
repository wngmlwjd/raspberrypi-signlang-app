from flask import Flask, render_template
import cv2
import numpy as np
import traceback

from config.config import CMD
from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite
from camera.camera_stream import CameraStream  # CameraStream 사용

app = Flask(__name__)

SEQ_LEN = 30
# buffer = deque(maxlen=SEQ_LEN)  # 필요시 활성화

# =========================
# CameraStream 설정
# =========================
cam = CameraStream(cmd=CMD)
print("✅ Camera stream started with rpicam-vid.")

# =========================
# Routes
# =========================
@app.route('/')
def index():
    frame_count = 0

    # Streaming generator
    def gen_frames():
        nonlocal frame_count
        while True:
            try:
                frame_bytes = cam.get_frame()
                if frame_bytes is None:
                    continue

                # JPEG → OpenCV 이미지
                frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

                # -------------------------------
                # 1) Extract hand landmarks & inference
                # -------------------------------
                landmarks = extract_landmarks(frame)
                # inference 코드는 필요시 여기에 추가

                # -------------------------------
                # 2) Encode frame as JPEG
                # -------------------------------
                ret, buffer_jpg = cv2.imencode('.jpg', frame)
                if not ret:
                    continue

                frame_bytes = buffer_jpg.tobytes()
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print("⚠️ Frame processing failed:", e)
                traceback.print_exc()
                continue

    # index.html에서 바로 MJPEG 스트리밍
    return render_template('index.html', gen_frames=gen_frames)

# =========================
# Run server
# =========================
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cam.stop()  # 서버 종료 시 카메라 스레드 정리
