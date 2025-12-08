from collections import deque
import numpy as np
import cv2
import time

from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

SEQ_LEN = 30
buffer = deque(maxlen=SEQ_LEN)

def run_realtime_inference():
    infer = AppInferenceTFLite()

    print("📸 Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera open failed.")
        return

    print("✅ Camera opened.")
    print("🔧 Real-time inference started...")
    print("-------------------------------------------")

    frame_count = 0
    last_print = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read failed... retrying")
            time.sleep(0.1)
            continue

        frame_count += 1
        now = time.time()

        # -------------------------------
        # 1) Extract hand landmarks
        # -------------------------------
        landmarks = extract_landmarks(frame)

        if landmarks is None:
            if now - last_print > 0.5:
                print("📌 No hand detected...")
                last_print = now
            continue
        else:
            if now - last_print > 0.5:
                print(f"🖐 Hand detected (frame {frame_count})")
                last_print = now

        # -------------------------------
        # 2) Preprocess → feature
        # -------------------------------
        feature = process_to_feature(landmarks)
        buffer.append(feature)

        # -------------------------------
        # 3) Buffer progress
        # -------------------------------
        filled = len(buffer)
        if now - last_print > 0.5:
            print(f"📚 Buffer: {filled}/{SEQ_LEN}")
            last_print = now

        # -------------------------------
        # 4) Inference
        # -------------------------------
        if filled == SEQ_LEN:
            print("\n🔮 Running inference...")
            seq_array = np.array(buffer)

            pred_word, pred_prob = infer.predict_from_array(seq_array)
            print(f"👉 Result: {pred_word}  |  confidence={pred_prob.max():.4f}")
            print("-------------------------------------------\n")

        # 라즈베리파이에서는 imshow 제거
        # q 입력은 콘솔에서 받을 수 없으므로 생략
        # 종료하려면 Ctrl + C

    cap.release()
    print("✨ Real-time inference stopped.")
