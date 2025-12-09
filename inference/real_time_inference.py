import subprocess
import cv2
import numpy as np
import time
from collections import deque
import os

from config.config import SEQUENCE_LENGTH, CMD, RAW_DIR
from inference.extract_frames import extract_frames
from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

buffer = deque(maxlen=SEQUENCE_LENGTH)

def rpicam_realtime_loop(interval=5):
    """
    interval: 영상 단위 녹화 시간 (초)
    """
    # infer = AppInferenceTFLite()
    file_index = 0

    while True:
        output_file = os.path.join(RAW_DIR, f"test_{file_index:04d}.mp4")
        file_index += 1

        # -------------------------------
        # 1) rpicam-vid로 영상 녹화
        # -------------------------------
        cmd = CMD + ["-t", str(interval*1000), "-o", output_file]
        print(f"📸 Recording video: {output_file} ...")
        subprocess.run(cmd)

        # -------------------------------
        # 2) 녹화된 영상에서 프레임 추출
        # -------------------------------
        print("🎞 Extracting frames...")
        frames = extract_frames(output_file)

        for frame_count, frame in enumerate(frames):
            # -------------------------------
            # 3) Landmark 추출
            # -------------------------------
            landmarks = extract_landmarks(frame)
            if landmarks is None:
                continue

            # -------------------------------
            # 4) Feature 전처리
            # -------------------------------
            feature = process_to_feature(landmarks)
            buffer.append(feature)

            # -------------------------------
            # 5) 버퍼 상태 및 Inference
            # -------------------------------
            # if len(buffer) == SEQUENCE_LENGTH:
            #     seq_array = np.array(buffer)
            #     pred_word, pred_prob = infer.predict_from_array(seq_array)
            #     print(f"👉 Result: {pred_word}  |  confidence={pred_prob.max():.4f}")
            #     print("-------------------------------------------")

        print(f"✅ Finished processing {output_file}\n")

if __name__ == "__main__":
    rpicam_realtime_loop(interval=5)
