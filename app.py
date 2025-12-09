# app.py
from flask import Flask, Response
from camera_stream import CameraStream
from config.config import CMD  # rpicam-vid 명령어

app = Flask(__name__)

camera = CameraStream(cmd=CMD)

def generate_frames():
    while True:
        frame_bytes = camera.get_frame_bytes()
        if frame_bytes is None:
            continue
        # MJPEG 스트림 형식으로 변환
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "<h1>Raspberry Pi Camera Stream</h1><img src='/video_feed'>"

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        camera.stop()
