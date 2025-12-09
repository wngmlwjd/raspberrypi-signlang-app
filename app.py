from flask import Flask, send_file, render_template
import subprocess
import time
import os

app = Flask(__name__)
CAM_FILE = "/tmp/cam.jpg"

def capture_image():
    """
    rpicam-vid 대신 rpicam-raw 이미지 캡처 명령어 사용
    """
    cmd = [
        "rpicam-raw",  # 단일 캡처용 명령어, 없으면 rpicam-vid로도 가능
        "-o", CAM_FILE,
        "--width", "640",
        "--height", "480",
        "--nopreview"
    ]
    subprocess.run(cmd)

@app.route("/snapshot")
def snapshot():
    capture_image()          # 새로고침 시마다 한 장 캡처
    # 파일 반환
    return send_file(CAM_FILE, mimetype="image/jpeg")

@app.route("/")
def index():
    # 브라우저에서 /snapshot 이미지를 보여줌
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
