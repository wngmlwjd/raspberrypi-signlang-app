# ui/routes.py
from flask import Blueprint, render_template, Response
from camera.camera_stream import CameraStream

bp = Blueprint("ui", __name__)

# 카메라 인스턴스 생성 (전역)
camera = CameraStream()

@bp.route("/")
def index():
    return render_template("index.html")

# MJPEG 스트리밍
def generate_stream():
    while True:
        frame = camera.get_frame()
        if frame:
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )

@bp.route("/video_feed")
def video_feed():
    return Response(
        generate_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
