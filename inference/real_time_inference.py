from collections import deque
import numpy as np
import cv2

from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import process_to_feature
from inference.TFLite import AppInferenceTFLite

SEQ_LEN = 30
buffer = deque(maxlen=SEQ_LEN)

def run_realtime_inference():
    infer = AppInferenceTFLite()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        landmarks = extract_landmarks(frame)
        if landmarks is None:
            continue

        feature = process_to_feature(landmarks)
        buffer.append(feature)

        if len(buffer) == SEQ_LEN:
            seq_array = np.array(buffer)
            pred_word, pred_prob = infer.predict_from_array(seq_array)
            print(pred_word, pred_prob.max())

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
