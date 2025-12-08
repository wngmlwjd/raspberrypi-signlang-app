# app.py
from flask import Flask, render_template, Response
from collections import deque
import cv2
import numpy as np
import time

from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

app = Flask(__name__)

SEQ_LEN = 30
buffer = deque(maxlen=SEQ_LEN)
infer = AppInferenceTFLite()

# =========================
# Video capture
# =========================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Camera open failed.")

# =========================
# Generator for streaming frames
# =========================
def gen_frames():
    frame_count = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        now = time.time()

        # -------------------------------
        # 1) Extract hand landmarks
        # -------------------------------
        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            feature = process_to_feature(landmarks)
            buffer.append(feature)

        # -------------------------------
        # 2) Inference
        # -------------------------------
        if len(buffer) == SEQ_LEN:
            seq_array = np.array(buffer)
            pred_word, pred_prob = infer.predict_from_array(seq_array)

            # 화면에 텍스트 오버레이
            cv2.putText(frame, f"{pred_word} ({pred_prob.max():.2f})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # -------------------------------
        # 3) Encode frame as JPEG
        # -------------------------------
        ret, buffer_jpg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer_jpg.tobytes()
        # multipart streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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
    app.run(host='0.0.0.0', port=5000, debug=False)
