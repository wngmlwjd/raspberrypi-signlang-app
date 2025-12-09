from flask import Flask, Response, render_template
from camera.camera_stream import CameraStream

app = Flask(__name__)

# 카메라 명령어
CAM_CMD = [
    "rpicam-vid",
    "-t", "0",
    "-o", "-",
    "--width", "640",
    "--height", "480",
    "--framerate", "30"
]

camera = CameraStream(cmd=CAM_CMD)

def gen_frames():
    while True:
        frame = camera.get_frame_bytes()
        if frame is None:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def index():
    return render_template("index.html")  # <img src="/video_feed">

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        camera.stop()
