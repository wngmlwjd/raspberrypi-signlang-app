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
    le_dict = joblib.load(path)
    log_message(f"Label encoder loaded (dict): {path}")
    return le_dict

# ----------------------------------------------------
# 3. 단일 feature 추론
# ----------------------------------------------------
def infer_feature(model, feature: np.ndarray) -> np.ndarray:
    """
    feature : (T, J_max*3)
    return  : softmax 확률
    """
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)
    pred = model.predict(input_data, verbose=0)
    return pred[0]

# ----------------------------------------------------
# 4. 단일 feature 전용 최종 라벨 + top5 출력
# ----------------------------------------------------
def infer_single_feature_with_top5(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
):
    model = load_h5_model(model_path)
    le_dict = load_label_encoder(label_encoder_path)

    # feature 1개만 사용
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")
    if len(feature_files) > 1:
        log_message("Warning: More than 1 feature found, using only the first one.")

    feature_path = os.path.join(features_dir, feature_files[0])
    feature = np.load(feature_path)

    # softmax 예측
    pred = infer_feature(model, feature)

    # top1
    top1_idx = np.argmax(pred)
    top1_label = le_dict['int_to_label'].get(top1_idx, "unknown")
    top1_prob = float(pred[top1_idx])

    # top5
    top5_idx = np.argsort(pred)[-5:][::-1]
    top5_labels = [le_dict['int_to_label'].get(i, "unknown") for i in top5_idx]
    top5_probs = [float(pred[i]) for i in top5_idx]

    return {
        "feature_file": feature_files[0],
        "pred_vector": pred,
        "top1_label": top1_label,
        "top1_prob": top1_prob,
        "top5_labels": top5_labels,
        "top5_probs": top5_probs
    }
