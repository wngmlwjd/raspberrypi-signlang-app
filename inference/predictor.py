import os
import numpy as np

from inference.TFLite import AppInferenceTFLite
from config.config import MODEL_PATH, FEATURES_DIR


def main():
    print("📌 Using config settings:")
    print(f"  - MODEL_PATH: {MODEL_PATH}")
    print(f"  - INPUT_FEATURE_DIR: {FEATURES_DIR}")

    # -----------------------------
    # 1) 모델, 인코더, maxJ 로드
    # -----------------------------
    print("\n📌 Loading model & encoder...")
    infer = AppInferenceTFLite()   # ← date 제거

    # -----------------------------
    # 2) FEATURES_DIR 내부 .npy 파일 수집
    # -----------------------------
    if not os.path.isdir(FEATURES_DIR):
        raise NotADirectoryError(f"입력 경로가 유효한 폴더가 아닙니다: {FEATURES_DIR}")

    npy_files = sorted([
        os.path.join(FEATURES_DIR, f)
        for f in os.listdir(FEATURES_DIR)
        if f.endswith(".npy")
    ])

    if len(npy_files) == 0:
        raise FileNotFoundError(f"폴더 내 .npy 파일이 없습니다: {FEATURES_DIR}")

    print(f"\n📁 Found {len(npy_files)} npy files.")
    print("-----------------------------")
    for i, f in enumerate(npy_files, 1):
        print(f"[{i}] {os.path.basename(f)}")
    print("-----------------------------\n")

    # -----------------------------
    # 3) 파일별 추론 실행
    # -----------------------------
    for idx, npy_path in enumerate(npy_files, 1):
        print(f"📌 Loading: {npy_path}")
        features = np.load(npy_path)

        print("📌 Running inference...")
        pred_word, pred_prob = infer.predict_from_array(features)

        # -----------------------------
        # 4) 결과 출력
        # -----------------------------
        print("\n======================")
        print(f"🟢 Prediction Result ({idx}/{len(npy_files)})")
        print("======================")
        print(f"File            : {os.path.basename(npy_path)}")
        print(f"Predicted Label : {pred_word}")
        print(f"Confidence      : {pred_prob.max():.4f}")
        print("======================\n")


if __name__ == "__main__":
    main()
