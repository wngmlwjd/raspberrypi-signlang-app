from collections import deque
import numpy as np
import cv2
import subprocess

from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

SEQ_LEN = 30
buffer = deque(maxlen=SEQ_LEN)


def run_realtime_inference():

    infer = AppInferenceTFLite()

    # ======== rpicam-vid → ffmpeg → OpenCV =========
    gst_pipeline = (
        "rpicam-vid --inline --nopreview -t 0 --width 640 --height 480 --framerate 30 "
        "-o - | "
        "ffmpeg -i pipe:0 -vcodec rawvideo -pix_fmt bgr24 -f rawvideo -"
    )

    print("📸 opening camera...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("❌ camera open failed")
        return

    print("✅ camera opened")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        print("ret:", ret)

        if not ret:
            continue

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"frames read: {frame_count}")

        # ======== Landmark 추출 ========
        landmarks = extract_landmarks(frame)
        if landmarks is None:
            continue

        # ======== Feature 변환 ========
        feature = process_to_feature(landmarks)
        buffer.append(feature)

        # ======== 추론 실행 ========
        if len(buffer) == SEQ_LEN:
            seq_array = np.array(buffer)
            pred_word, pred_prob = infer.predict_from_array(seq_array)
            print(f"[PRED] {pred_word}: {pred_prob.max():.3f}")

        # ======== 화면 표시 ========
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
