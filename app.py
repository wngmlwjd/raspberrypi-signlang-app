from flask import Flask, render_template, jsonify, send_file
import threading
from inference import video_saver  # save_video()가 있는 모듈
from config import config  # config.VIDEO_PATH 사용
import os

app = Flask(__name__)

recording_thread = None
recording_status = "대기 중"  # 녹화 상태 저장

def record_video():
    global recording_status
    recording_status = "녹화 중..."
    video_saver.save_video()  # 실제 녹화 함수
    recording_status = "녹화 완료"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread
    if recording_thread is None or not recording_thread.is_alive():
        recording_thread = threading.Thread(target=record_video, daemon=True)
        recording_thread.start()
        return jsonify({"status": "녹화를 시작했습니다!"})
    else:
        return jsonify({"status": "이미 녹화가 진행 중입니다."})

@app.route("/recording_status")
def get_status():
    return jsonify({"status": recording_status})

@app.route("/recorded_video")
def recorded_video():
    # 녹화된 파일이 존재하면 브라우저로 전송
    if os.path.exists(config.VIDEO_PATH):
        return send_file(config.VIDEO_PATH, mimetype="video/mp4")
    return "영상이 없습니다.", 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
