from flask import Flask, Response, render_template
import subprocess

app = Flask(__name__)

def generate():
    """
    rpicam-vid를 subprocess로 실행하고 stdout에서 MJPEG 프레임 읽기
    """
    cmd = [
        "rpicam-vid",
        "-t", "0",
        "-o", "-",
        "--width", "160",    # 1/4 크기로 줄임
        "--height", "120",   # 1/4 크기로 줄임
        "--framerate", "30",
        "--codec", "mjpeg"
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8) as proc:
        while True:
            frame = proc.stdout.read(1024)
            if not frame:
                break
            yield frame

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=--frame')

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
