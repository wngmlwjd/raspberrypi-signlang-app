import os
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from collections import Counter

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
# top5 합산 대신 확률 가중 평균 기반 최종 라벨 결정
# ----------------------------------------------------
def predict_weighted_average(all_preds, le_dict):
    """
    all_preds : (num_features, num_classes) - 각 feature별 softmax
    le_dict   : int_to_label dict
    return    : 최종 예측 라벨
    """
    # feature별 확률 평균 계산
    avg_probs = np.mean(all_preds, axis=0)  # (num_classes,)
    
    # 확률 최대인 클래스 선택
    final_idx = np.argmax(avg_probs)
    final_label = le_dict['int_to_label'].get(final_idx, "unknown")
    return final_label


# ----------------------------------------------------
# 폴더 내 feature 전체 추론 + 라벨 디코딩 (수정)
# ----------------------------------------------------
def infer_features_in_dir(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
    use_weighted_average: bool = True  # True이면 확률 기반, False이면 기존 top5 다수결
):
    model = load_h5_model(model_path)
    le_dict = load_label_encoder(label_encoder_path)

    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")

    all_preds = []
    feature_labels = []
    top5_per_feature = []

    for f in feature_files:
        feature = np.load(os.path.join(features_dir, f))
        pred = infer_feature(model, feature)

        # 단일 feature 최종 라벨
        label_idx = np.argmax(pred)
        label_name = le_dict['int_to_label'].get(label_idx, "unknown")
        feature_labels.append(label_name)

        # top5 라벨
        top5_idx = np.argsort(pred)[-5:][::-1]
        top5_labels = [le_dict['int_to_label'].get(i, "unknown") for i in top5_idx]
        top5_per_feature.append(top5_labels)

        all_preds.append(pred)

    all_preds = np.array(all_preds)

    # 최종 라벨 결정
    if use_weighted_average:
        final_label = predict_weighted_average(all_preds, le_dict)
    else:
        from collections import Counter
        # 기존 top5 다수결 방식
        top5_labels_flat = [label for sublist in top5_per_feature for label in sublist]
        counter = Counter(top5_labels_flat)
        final_label = counter.most_common(1)[0][0]

    return all_preds, feature_labels, top5_per_feature, final_label
