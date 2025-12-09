import subprocess
import cv2
import numpy as np
import time
from collections import deque

from config.config import SEQUENCE_LENGTH, CMD
from inference.extract_frames import extract_frames
from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

buffer = deque(maxlen=SEQUENCE_LENGTH)

def rpicam_realtime_inference():
    # -------------------------------
    # rpicam-vid 명령어 설정
    # -------------------------------

    proc = subprocess.Popen(CMD, stdout=subprocess.PIPE, bufsize=10**8)

    infer = AppInferenceTFLite()
    print("📸 Camera stream started...")

    data = b""
    frame_count = 0
    last_print = time.time()

    while True:
        # stdout에서 데이터 읽기
        chunk = proc.stdout.read(1024)
        if not chunk:
            break
        data += chunk

        # JPEG 프레임 단위 분리
        start = data.find(b'\xff\xd8')
        end = data.find(b'\xff\xd9')
        if start != -1 and end != -1:
            jpg = data[start:end+2]
            data = data[end+2:]

            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                # -------------------------------
                # 1) 프레임 저장 (선택 사항)
                # -------------------------------
                cv2.imwrite(f"frame_{frame_count:04d}.jpg", frame)
                frame_count += 1

                # -------------------------------
                # 2) Extract landmarks
                # -------------------------------
                landmarks = extract_landmarks(frame)
                if landmarks is None:
                    now = time.time()
                    if now - last_print > 0.5:
                        print("📌 No hand detected...")
                        last_print = now
                    continue

                # -------------------------------
                # 3) Feature 전처리
                # -------------------------------
                feature = process_to_feature(landmarks)
                buffer.append(feature)

                # -------------------------------
                # 4) 버퍼 상태 출력
                # -------------------------------
                now = time.time()
                filled = len(buffer)
                if now - last_print > 0.5:
                    print(f"📚 Buffer: {filled}/{SEQUENCE_LENGTH}")
                    last_print = now

                # -------------------------------
                # 5) Inference
                # -------------------------------
                if filled == SEQUENCE_LENGTH:
                    seq_array = np.array(buffer)
                    pred_word, pred_prob = infer.predict_from_array(seq_array)
                    print(f"👉 Result: {pred_word}  |  confidence={pred_prob.max():.4f}")
                    print("-------------------------------------------")

        # 종료 조건: Ctrl + C 로 강제 종료

    proc.terminate()
    print("✨ Real-time inference stopped.")

if __name__ == "__main__":
    rpicam_realtime_inference()
