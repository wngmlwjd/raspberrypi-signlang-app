from flask import Flask, render_template, jsonify, send_file
import threading
import os
import shutil

from inference.video_saver import save_video
from inference.extract_frames import extract_frames
from inference.extract_landmarks import extract_landmarks
from config import config

app = Flask(__name__)

recording_thread = None
recording_status = "대기 중"
frame_count = 0

def record_video():
    global recording_status, frame_count
    recording_status = "녹화 중..."
    save_video()  # 녹화 실행
    recording_status = "녹화 완료. 프레임 추출 중..."
    
    # 프레임 추출
    frame_count = extract_frames(output_dir=config.FRAMES_DIR)
    recording_status = f"프레임 추출 완료 ({frame_count} 프레임)"  # ← 여기 수정

    # 랜드마크 추출
    recording_status = "랜드마크 추출 중..."
    extract_landmarks(frame_dir=config.FRAMES_DIR, save_dir=config.LANDMARKS_DIR)

    # 최종 상태
    recording_status = f"녹화, 프레임 및 랜드마크 추출 완료 ({frame_count} 프레임)"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread
    if recording_thread is None or not recording_thread.is_alive():

        # 기존 frames 폴더 삭제 후 재생성
        if os.path.exists(config.FRAMES_DIR):
            shutil.rmtree(config.FRAMES_DIR)
        os.makedirs(config.FRAMES_DIR, exist_ok=True)

        # 기존 landmarks 폴더 삭제 후 재생성
        if os.path.exists(config.LANDMARKS_DIR):
            shutil.rmtree(config.LANDMARKS_DIR)
        os.makedirs(config.LANDMARKS_DIR, exist_ok=True)

        recording_thread = threading.Thread(target=record_video, daemon=True)
        recording_thread.start()
        return jsonify({"status": "녹화를 시작했습니다!"})
    else:
        return jsonify({"status": "이미 녹화가 진행 중입니다."})

@app.route("/recording_status")
def get_status():
    return jsonify({"status": recording_status, "frame_count": frame_count})

@app.route("/recorded_video")
def recorded_video():
    if os.path.exists(config.VIDEO_PATH):
        return send_file(config.VIDEO_PATH, mimetype="video/mp4")
    return "영상이 없습니다.", 404

@app.route("/frames/<filename>")
def serve_frame(filename):
    path = os.path.join(config.FRAMES_DIR, filename)  # <-- 변경
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "프레임이 없습니다.", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
