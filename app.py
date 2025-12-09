from flask import Flask, render_template, jsonify, send_from_directory
import threading
import os
from inference import video_saver  # save_video()가 있는 모듈
from config.config import RAW_DIR, VIDEO_PATH

app = Flask(__name__)

recording_thread = None

def record_video():
    video_saver.save_video()  # 파일 이름 반환하도록 save_video 수정 필요

@app.route("/")
def index():
    return render_template("index.html", latest_video=VIDEO_PATH)

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread
    if recording_thread is None or not recording_thread.is_alive():
        recording_thread = threading.Thread(target=record_video, daemon=True)
        recording_thread.start()
        return jsonify({"status": "녹화 시작됨"})
    else:
        return jsonify({"status": "이미 녹화 중"})

@app.route("/recorded/<filename>")
def recorded_video(filename):
    return send_from_directory(RAW_DIR, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
