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
    cap = cv2.VideoCapture(0)

    frame_count = 0
    last_print_time = time.time()

    print("🔧 Real-time inference started...")
    print("-------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Frame read failed, skipping...")
            continue

        frame_count += 1

        # =========================
        # 랜드마크 추출
        # =========================
        landmarks = extract_landmarks(frame)

        if landmarks is None:
            if time.time() - last_print_time > 1.0:
                print("📌 No hand detected...")
                last_print_time = time.time()
            continue
        else:
            print(f"🖐 Hand detected (frame {frame_count})")

        # =========================
        # 전처리 → feature 생성
        # =========================
        feature = process_to_feature(landmarks)
        buffer.append(feature)

        # 버퍼 진행 상황 출력
        filled = len(buffer)
        print(f"📚 Buffer: {filled}/{SEQ_LEN}")

        # =========================
        # 추론 수행
        # =========================
        if filled == SEQ_LEN:
            print("\n🔮 Running inference...")
            seq_array = np.array(buffer)

            pred_word, pred_prob = infer.predict_from_array(seq_array)
            print(f"👉 Result: {pred_word}  |  confidence={pred_prob.max():.4f}")
            print("-------------------------------------------\n")

        # =========================
        # 화면 표시
        # =========================
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            print("🛑 Stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✨ Real-time inference stopped.")
