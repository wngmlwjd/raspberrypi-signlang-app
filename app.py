# app.py
from flask import Flask, render_template, Response
from collections import deque
import cv2
import numpy as np
import time
import traceback

from config.config import CMD
from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite
from camera.camera_stream import CameraStream  # 새로 만든 클래스

app = Flask(__name__)

# try:
#     infer = AppInferenceTFLite()
# except Exception as e:
#     print("❌ Failed to load TFLite model:", e)
#     traceback.print_exc()
#     infer = None

# =========================
# CameraStream 설정
# =========================
cam = CameraStream(cmd=CMD)
print("✅ Camera stream started with rpicam-vid.")

# =========================
# Generator for streaming frames
# =========================
def gen_frames():
    frame_count = 0
    last_print = time.time()

    while True:
        try:
            frame_bytes = cam.get_frame()
            if frame_bytes is None:
                continue

            frame_count += 1

            # JPEG → OpenCV 이미지
            frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

            # -------------------------------
            # 1) Extract hand landmarks & inference
            # -------------------------------
            landmarks = extract_landmarks(frame)
            # if landmarks is not None and infer is not None:
            #     try:
            #         feature = process_to_feature(landmarks)
            #         buffer.append(feature)

            #         if len(buffer) == SEQ_LEN:
            #             seq_array = np.array(buffer)
            #             pred_word, pred_prob = infer.predict_from_array(seq_array)

            #             # 화면에 텍스트 오버레이
            #             cv2.putText(frame, f"{pred_word} ({pred_prob.max():.2f})",
            #                         (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            #     except Exception as e:
            #         print("⚠️ Inference failed:", e)
            #         traceback.print_exc()

            # # -------------------------------
            # # 2) Encode frame as JPEG
            # # -------------------------------
            # ret, buffer_jpg = cv2.imencode('.jpg', frame)
            # if not ret:
            #     continue

            # frame_bytes = buffer_jpg.tobytes()
            # yield (b'--frame\r\n'
            #        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            print("⚠️ Frame processing failed:", e)
            traceback.print_exc()
            continue

# =========================
# Routes
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# =========================
# Run server
# =========================
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        cam.stop()  # 서버 종료 시 카메라 스레드 정리
