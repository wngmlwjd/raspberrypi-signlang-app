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
    """
    MJPEG 파일을 읽어 브라우저에 스트리밍
    """
    while True:
        if not os.path.exists(CAM_FILE):
            time.sleep(0.1)
            continue
        
        with open(CAM_FILE, "rb") as f:
            chunk = f.read()
            if chunk:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n"

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=5000, debug=True)
