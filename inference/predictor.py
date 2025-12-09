# inference/predictor_h5.py
import os
import numpy as np
from tensorflow.keras.models import load_model

from config.config import FEATURES_DIR, MODEL_PATH
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
# 2. 단일 feature 추론
# ----------------------------------------------------
def infer_feature(model, feature: np.ndarray) -> np.ndarray:
    """
    feature : (T, J_max*3)
    return  : 모델 예측 결과
    """
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)  # batch dimension
    pred = model.predict(input_data, verbose=0)
    return pred[0]

# ----------------------------------------------------
# 3. 폴더 내 feature 전체 추론
# ----------------------------------------------------
def infer_features_in_dir(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH
):
    model = load_h5_model(model_path)
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")

    all_preds = []
    for f in feature_files:
        feature = np.load(os.path.join(features_dir, f))
        pred = infer_feature(model, feature)
        log_message(f"Inferred {f}: {pred}")
        all_preds.append(pred)

    return np.array(all_preds)

