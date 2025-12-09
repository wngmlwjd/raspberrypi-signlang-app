from flask import Flask, render_template, jsonify
import threading
from inference import video_saver  # save_video()가 있는 모듈
from config import config  # config.VIDEO_PATH 사용

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
        # 녹화 실행 스레드 생성
        recording_thread = threading.Thread(target=record_video, daemon=True)
        recording_thread.start()
        return jsonify({"status": "녹화를 시작했습니다!", "video_path": config.VIDEO_PATH})
    else:
        return jsonify({"status": "이미 녹화가 진행 중입니다.", "video_path": config.VIDEO_PATH})

@app.route("/recording_status")
def get_status():
    # 현재 녹화 상태와 영상 경로 반환
    return jsonify({"status": recording_status, "video_path": config.VIDEO_PATH})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
