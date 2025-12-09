from flask import Flask, Response, render_template
import subprocess
import time
import os

app = Flask(__name__)

CAM_FILE = "/dev/shm/cam.mjpeg"

def start_camera():
    """
    rpicam-vid를 백그라운드에서 실행하고 MJPEG 파일로 출력
    """
    # 기존에 실행 중인 프로세스 종료
    subprocess.call(["pkill", "-f", "rpicam-vid"])
    
    cmd = [
        "rpicam-vid",
        "--nopreview",       # Preview 창 없이 실행
        "-t", "0",           # 무한 실행
        "-o", CAM_FILE,      # MJPEG 파일로 출력
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg"
    ]
    subprocess.Popen(cmd)

def generate():
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "-o", "-",
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg",
        "--nopreview"
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8) as proc:
        data = b""
        while True:
            chunk = proc.stdout.read(1024)
            if not chunk:
                break
            data += chunk
            # JPEG 프레임 시작/끝 인식
            while b"\xff\xd8" in data and b"\xff\xd9" in data:
                start = data.find(b"\xff\xd8")
                end = data.find(b"\xff\xd9") + 2
                frame = data[start:end]
                data = data[end:]
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=5000, debug=True)
