from flask import Flask, Response, render_template
from camera.camera_stream import CameraStream
from config.config import CMD

app = Flask(__name__)

# 카메라 스트림 시작
camera = CameraStream(cmd=CMD)

def generate():
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        camera.stop()