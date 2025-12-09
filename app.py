from flask import Flask, Response, render_template
import subprocess
import time

app = Flask(__name__)

# rpicam-vid를 MJPEG 파일로 출력
CAM_FILE = "/dev/shm/cam.mjpeg"

def start_camera():
    cmd = [
        "rpicam-vid",
        "--nopreview",
        "-t", "0",
        "-o", CAM_FILE,
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--codec", "mjpeg"
    ]
    # 백그라운드로 실행
    subprocess.Popen(cmd)

def generate():
    """MJPEG 파일을 읽어 프레임 단위로 전송"""
    while True:
        try:
            with open(CAM_FILE, "rb") as f:
                while True:
                    chunk = f.read(1024)
                    if not chunk:
                        break
                    yield chunk
        except FileNotFoundError:
            time.sleep(0.1)

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=--frame')

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    start_camera()
    app.run(host="0.0.0.0", port=5000, debug=True)
