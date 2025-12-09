from flask import Flask, render_template, jsonify, send_file
import threading
import os
import shutil

from inference.video_saver import save_video
from inference.extract_frames import extract_frames
from inference.extract_landmarks import extract_landmarks
from inference.preprocessor import generate_single_feature
from inference.predictor import infer_single_feature_with_top3  # ← 새로 작성한 곳
from config import config
from utils import log_message

app = Flask(__name__)

# -----------------------------
# 전역 변수
# -----------------------------
recording_thread = None
recording_status = "대기 중"
frame_count = 0
landmark_count = 0
feature_count = 0

predictions = None           # softmax 벡터 1개
predicted_labels = None      # top1 + top5 정보


# -----------------------------
# 녹화 및 처리 함수
# -----------------------------
def record_video():
    global recording_status, frame_count, landmark_count, feature_count
    global predictions, predicted_labels

    recording_status = "녹화 중..."

    # 1) 영상 녹화
    save_video()
    recording_status = "녹화 완료. 프레임 추출 중..."

    # 2) 프레임 추출
    frame_count = extract_frames()
    recording_status = "프레임 추출 완료. 랜드마크 추출 중..."

    # 3) 랜드마크 추출
    landmark_count = extract_landmarks()

    # 4) feature 생성 (슬라이딩 결과 1개만 생성된다고 가정)
    feature_count = generate_single_feature()
    recording_status = "랜드마크 처리 완료. 추론 중..."

    # 5) 단일 feature 추론
    result = infer_single_feature_with_top3()

    predictions = result["pred_vector"]
    predicted_labels = {
        "feature_file": result["feature_file"],
        "top1_label": result["top1_label"],
        "top1_prob": round(result["top1_prob"], 4),
        "top3_labels": result["top3_labels"],
        "top3_probs": [round(p, 4) for p in result["top3_probs"]],
    }

    recording_status = f"전체 프로세스 완료. feature 1개 생성됨"
    log_message(f"예측 완료: top1 = {predicted_labels['top1_label']} "
                f"({predicted_labels['top1_prob']})")


# -----------------------------
# 라우팅
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread

    if recording_thread is None or not recording_thread.is_alive():

        # 기존 폴더 초기화
        for folder in [config.FRAMES_DIR, config.LANDMARKS_DIR,
                       config.DRAW_LANDMARKS_DIR, config.FEATURES_DIR]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)

        # 녹화 스레드 시작
        recording_thread = threading.Thread(target=record_video, daemon=True)
        recording_thread.start()

        return jsonify({"status": "녹화를 시작했습니다!"})
    else:
        return jsonify({"status": "이미 녹화가 진행 중입니다."})


@app.route("/recording_status")
def get_status():
    return jsonify({
        "status": recording_status,
        "frame_count": frame_count,
        "landmark_count": landmark_count,
        "feature_count": feature_count,
        "predictions_shape": None if predictions is None else len(predictions),
        "predicted_labels": predicted_labels
    })


@app.route("/recorded_video")
def recorded_video():
    if os.path.exists(config.VIDEO_PATH):
        return send_file(config.VIDEO_PATH, mimetype="video/mp4")
    return "영상이 없습니다.", 404


@app.route("/frames/<filename>")
def serve_frame(filename):
    path = os.path.join(config.FRAMES_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "프레임이 없습니다.", 404


@app.route("/draw_landmarks/<filename>")
def serve_draw_landmark(filename):
    path = os.path.join(config.DRAW_LANDMARKS_DIR, filename)
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "랜드마크 이미지가 없습니다.", 404


# -----------------------------
# 앱 실행
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
