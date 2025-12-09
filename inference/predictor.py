import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib

from config.config import FEATURES_DIR, MODEL_PATH, LABEL_ENCODER_PATH
from utils import log_message

# ----------------------------------------------------
# 1. .h5 모델 로드
# ----------------------------------------------------
def load_h5_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"H5 model not found: {model_path}")
    model = load_model(model_path)
    log_message(f".h5 model loaded: {model_path}")
    return model

# ----------------------------------------------------
# 2. label encoder 로드 (dict)
# ----------------------------------------------------
def load_label_encoder(path: str = LABEL_ENCODER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label encoder not found: {path}")
    le_dict = joblib.load(path)  # {index: label} 형태라고 가정
    log_message(f"Label encoder loaded (dict): {path}")
    return le_dict

# ----------------------------------------------------
# 3. 단일 feature 추론
# ----------------------------------------------------
def infer_feature(model, feature: np.ndarray) -> np.ndarray:
    """
    feature : (T, J_max*3)
    return  : 모델 예측 결과 (softmax 확률)
    """
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)  # batch dimension
    pred = model.predict(input_data, verbose=0)
    return pred[0]

# ----------------------------------------------------
# 4. 폴더 내 feature 전체 추론 + 라벨 디코딩
# ----------------------------------------------------
def infer_features_in_dir(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH
):
    model = load_h5_model(model_path)
    le_dict = load_label_encoder(label_encoder_path)

    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")

    all_preds = []
    all_labels = []

    for f in feature_files:
        feature = np.load(os.path.join(features_dir, f))
        pred = infer_feature(model, feature)
        label_idx = np.argmax(pred)
        label_name = le_dict.get(label_idx, "Unknown")  # dict lookup
        log_message(f"Inferred {f}: {label_name} (prob={pred[label_idx]:.3f})")
        all_preds.append(pred)
        all_labels.append(label_name)

    return np.array(all_preds), all_labels
