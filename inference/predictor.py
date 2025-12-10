import os
import numpy as np
import joblib
import tensorflow as tf

from config.config import FEATURES_DIR, MODEL_PATH, LABEL_ENCODER_PATH
from utils import log_message

# ----------------------------------------------------
# 1. TFLite 모델 로드
# ----------------------------------------------------
def load_tflite_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    log_message(f"TFLite model loaded: {model_path}")
    return interpreter

# ----------------------------------------------------
# 2. label encoder 로드
# ----------------------------------------------------
def load_label_encoder(path: str = LABEL_ENCODER_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Label encoder not found: {path}")
    le_dict = joblib.load(path)
    log_message(f"Label encoder loaded (dict): {path}")
    return le_dict

# ----------------------------------------------------
# 3. 단일 feature 추론 (TFLite)
# ----------------------------------------------------
def infer_feature(interpreter, feature: np.ndarray) -> np.ndarray:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # (1, T, F) 형태로 확장
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)

    # 입력 텐서 설정
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # 추론 실행
    interpreter.invoke()

    # 출력 벡터 가져오기
    pred = interpreter.get_tensor(output_details[0]['index'])[0]

    return pred

# ----------------------------------------------------
# 4. 단일 feature 최종 라벨 + top3 출력
# ----------------------------------------------------
def infer_single_feature_with_top3(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH,
    label_encoder_path: str = LABEL_ENCODER_PATH,
):
    interpreter = load_tflite_model(model_path)
    le_dict = load_label_encoder(label_encoder_path)

    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")
    if len(feature_files) > 1:
        log_message("Warning: More than 1 feature found, using only the first one.")

    feature_path = os.path.join(features_dir, feature_files[0])
    feature = np.load(feature_path)

    pred = infer_feature(interpreter, feature)

    # top1
    top1_idx = np.argmax(pred)
    top1_label = le_dict['int_to_label'].get(top1_idx, "unknown")
    top1_prob = float(pred[top1_idx])

    # top3
    top3_idx = np.argsort(pred)[-3:][::-1]
    top3_labels = [le_dict['int_to_label'].get(i, "unknown") for i in top3_idx]
    top3_probs = [float(pred[i]) for i in top3_idx]

    return {
        "feature_file": feature_files[0],
        "pred_vector": pred,
        "top1_label": top1_label,
        "top1_prob": top1_prob,
        "top3_labels": top3_labels,
        "top3_probs": top3_probs
    }
