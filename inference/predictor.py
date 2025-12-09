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
# 4-2. 격노 확률 임계치 기반 최종 라벨 결정
# ----------------------------------------------------
def predict_ignore_kyukno(all_preds, le_dict, kyukno_label="격노", threshold=0.95, recent_n=5, smoothing=0.01):
    """
    all_preds : (num_features, num_classes) - 각 feature별 softmax
    kyukno_label : '격노' 라벨 이름
    threshold    : 격노 확률이 이 값 이상이면 격노 유지
    recent_n     : 최근 N개 feature만 반영
    smoothing    : 확률 안정화용 최소값
    return       : 최종 예측 라벨, 최종 확률
    """
    recent_preds = all_preds[-recent_n:]
    avg_probs = np.clip(np.mean(recent_preds, axis=0), smoothing, 1.0)
    avg_probs /= avg_probs.sum()  # 정규화

    # '격노' index 찾기
    kyukno_idx = None
    for k, v in le_dict['int_to_label'].items():
        if v == kyukno_label:
            kyukno_idx = k
            break

    # 격노 확률 확인
    if kyukno_idx is not None and avg_probs[kyukno_idx] < threshold:
        avg_probs[kyukno_idx] = 0  # 격노 제외
        avg_probs /= avg_probs.sum()  # 재정규화

    final_idx = np.argmax(avg_probs)
    final_label = le_dict['int_to_label'].get(final_idx, "unknown")
    final_prob = float(avg_probs[final_idx])

    return final_label, final_prob

# ----------------------------------------------------
# 5. 폴더 내 feature 전체 추론 + Top5 확률 개선 + 격노 제외
# ----------------------------------------------------
def infer_features_in_dir_realistic_kyukno(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
    use_weighted_average: bool = True,
    recent_n: int = 5,
    kyukno_label: str = "격노",
    kyukno_threshold: float = 0.95
):
    model = load_h5_model(model_path)
    le_dict = load_label_encoder(label_encoder_path)

    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")

    all_preds = []
    feature_labels = []
    top5_per_feature = []
    top5_probs_per_feature = []

    for f in feature_files:
        feature = np.load(os.path.join(features_dir, f))
        pred = infer_feature(model, feature)

        # 단일 feature top1 라벨
        label_idx = np.argmax(pred)
        label_name = le_dict['int_to_label'].get(label_idx, "unknown")
        feature_labels.append(label_name)

        # top5 라벨 및 확률
        top5_idx = np.argsort(pred)[-5:][::-1]
        top5_labels = [le_dict['int_to_label'].get(i, "unknown") for i in top5_idx]
        top5_per_feature.append(top5_labels)
        top5_probs_per_feature.append([float(pred[i]) for i in top5_idx])

        all_preds.append(pred)

    all_preds = np.array(all_preds)

    if use_weighted_average:
        final_label, final_prob = predict_ignore_kyukno(
            all_preds,
            le_dict,
            kyukno_label=kyukno_label,
            threshold=kyukno_threshold,
            recent_n=recent_n
        )
    else:
        from collections import Counter
        top5_labels_flat = [label for sublist in top5_per_feature for label in sublist]
        counter = Counter(top5_labels_flat)
        final_label = counter.most_common(1)[0][0]
        final_prob = None

    return all_preds, feature_labels, top5_per_feature, top5_probs_per_feature, final_label, final_prob
