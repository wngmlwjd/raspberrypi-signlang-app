import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_DIR = os.path.join(BASE_DIR, "dataset")
FRAMES_DIR = os.path.join(DATASET_DIR, "frames")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
FEATURES_DIR = os.path.join(DATASET_DIR, "features")

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.tflite")
MAXJ_PATH = os.path.join(MODEL_DIR, "max_joints.txt")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

VIDEO_PATH = "/Users/wngmlwjd/workspace/github/raspberrypi-signlang-app/dataset/raw/NIA_SL_WORD1501_REAL01_D.mp4"

# ===============================
# MediaPipe Hands 설정
# ===============================
MEDIAPIPE_HANDS_CONFIG = {
    "static_image_mode": False,
    "max_num_hands": 2,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

SEQUENCE_LENGTH = 30  # 모델 입력 시퀀스 길이
SEQUENCE_STEP = 5    # 시퀀스 생성 시 프레임 이동 간격