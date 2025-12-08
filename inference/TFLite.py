import os
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import tflite_runtime.interpreter as tflite

from config.config import MODEL_PATH, LABEL_ENCODER_PATH, MAXJ_PATH


# =====================================================
# Label Encoder Loader (테스트 코드와 동일)
# =====================================================
def load_label_encoder_safely(path):
    enc = joblib.load(path)

    if isinstance(enc, LabelEncoder):
        return enc

    if isinstance(enc, dict):
        if "classes" in enc:
            le = LabelEncoder()
            le.classes_ = np.array(enc["classes"])
            return le

        if "label_to_int" in enc and "int_to_label" in enc:
            labels = [enc["int_to_label"][i] for i in sorted(enc["int_to_label"].keys())]
            le = LabelEncoder()
            le.fit(labels)
            return le

    raise ValueError("Unsupported label encoder format.")


# =====================================================
# maxJ 처리
# =====================================================
def compute_target_dim(feature_array, max_joints=None):
    if max_joints is None:
        max_joints = feature_array.shape[1] // 3

    target_dim = max_joints * 3
    return max_joints, target_dim


def pad_or_cut_features(features, target_dim):
    seq_len, dim = features.shape

    if dim < target_dim:
        features = np.pad(features, ((0, 0), (0, target_dim - dim)), mode='constant')

    elif dim > target_dim:
        features = features[:, :target_dim]

    return features


# =====================================================
#   TFLite Interpreter 기반 추론 클래스 (config 기반)
# =====================================================
class AppInferenceTFLite:
    def __init__(self):
        # -----------------------------
        # Load Label Encoder
        # -----------------------------
        self.label_encoder = load_label_encoder_safely(LABEL_ENCODER_PATH)

        # -----------------------------
        # Load maxJ
        # -----------------------------
        if os.path.exists(MAXJ_PATH):
            with open(MAXJ_PATH, "r") as f:
                self.max_joints = int(f.read().strip())
        else:
            self.max_joints = None

        # -----------------------------
        # Load TFLite model (Flex delegate 제거)
        # -----------------------------
        self.tflite_path = MODEL_PATH
        print(f"📌 Loading TFLite model: {self.tflite_path}")

        try:
            self.interpreter = tflite.Interpreter(model_path=self.tflite_path)
            self.interpreter.allocate_tensors()
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load TFLite model: {e}")

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # shape: (1, T, D)
        self.expected_seq_len = self.input_details[0]["shape"][1]
        self.expected_dim = self.input_details[0]["shape"][2]

    # --------------------------------------------------
    # 1 sequence array 입력 → 예측
    # --------------------------------------------------
    def predict_from_array(self, seq_array):
        """
        seq_array: (T, dim)
        """

        # maxJ 기반 target_dim reshape
        max_joints, target_dim = compute_target_dim(seq_array, self.max_joints)

        # dim 기준 pad/cut
        seq_array = pad_or_cut_features(seq_array, target_dim)

        # -----------------------------
        # 시간축(T) pad/cut
        # -----------------------------
        T = seq_array.shape[0]

        if T < self.expected_seq_len:
            pad_len = self.expected_seq_len - T
            seq_array = np.pad(seq_array, ((0, pad_len), (0, 0)), mode='constant')

        elif T > self.expected_seq_len:
            seq_array = seq_array[:self.expected_seq_len, :]

        # 배치 차원 추가 → float32
        input_data = np.expand_dims(seq_array.astype(np.float32), axis=0)

        # -----------------------------
        # TFLite 추론
        # -----------------------------
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        pred_prob = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        pred_idx = np.argmax(pred_prob)
        pred_word = self.label_encoder.inverse_transform([pred_idx])[0]

        return pred_word, pred_prob

    # --------------------------------------------------
    # npy 파일에서 직접 예측
    # --------------------------------------------------
    def predict_from_file(self, npy_path):
        seq_array = np.load(npy_path)
        return self.predict_from_array(seq_array)
