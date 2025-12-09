from flask import Flask, render_template
import threading
from inference import video_saver  # save_video()가 있는 모듈

app = Flask(__name__)

recording_thread = None

@app.route("/")
def index():
    # 버튼 포함 페이지 렌더링
    return render_template("index.html")

@app.route("/start_recording", methods=["POST"])
def start_recording():
    global recording_thread
    if recording_thread is None or not recording_thread.is_alive():
        # args 없이 save_video() 실행
        recording_thread = threading.Thread(target=video_saver.save_video, daemon=True)
        recording_thread.start()
        return "녹화를 시작했습니다!"
    else:
        return "이미 녹화가 진행 중입니다."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
