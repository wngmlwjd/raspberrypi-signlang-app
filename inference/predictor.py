import os
import numpy as np
import tensorflow as tf

from config.config import FEATURES_DIR, MODEL_PATH
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
# 2. 단일 feature 추론
# ----------------------------------------------------
def infer_feature(interpreter, feature: np.ndarray) -> np.ndarray:
    """
    feature : (T, J_max*3)
    return  : 모델 예측 결과
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # TFLite는 batch dimension 필요
    input_data = np.expand_dims(feature, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # batch 제거

# ----------------------------------------------------
# 3. 폴더 내 feature 전체 추론
# ----------------------------------------------------
def infer_features_in_dir(
    features_dir: str = FEATURES_DIR,
    model_path: str = MODEL_PATH
):
    interpreter = load_tflite_model(model_path)
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".npy")])
    if not feature_files:
        raise FileNotFoundError(f"No feature files found: {features_dir}")

    all_preds = []
    for f in feature_files:
        feature = np.load(os.path.join(features_dir, f))
        pred = infer_feature(interpreter, feature)
        log_message(f"Inferred {f}: {pred}")
        all_preds.append(pred)

    return np.array(all_preds)

# ----------------------------------------------------
# 4. 실행 예제
# ----------------------------------------------------
if __name__ == "__main__":
    predictions = infer_features_in_dir()
    print("All predictions shape:", predictions.shape)
