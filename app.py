# app_h264.py
from flask import Flask, Response
import subprocess
import threading
import cv2
import numpy as np
import time

app = Flask(__name__)

class CameraH264:
    def __init__(self, cmd=None):
        """
        cmd : list
            rpicam-vid H264 스트림 명령어
        """
        if cmd is None:
            raise ValueError("rpicam-vid 명령어를 cmd 인자로 전달해야 합니다.")

        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update_stream, daemon=True)
        self.thread.start()

    def _update_stream(self):
        """
        H264 스트림을 OpenCV에서 읽을 수 있도록 변환 (디코딩)
        """
        cap = cv2.VideoCapture(self.proc.stdout)  # stdout에서 H264 읽기
        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame_bytes(self):
        """
        웹 스트리밍용 JPEG 바이트 반환
        """
        if self.frame is None:
            return None
        ret, jpeg = cv2.imencode(".jpg", self.frame)
        if not ret:
            return None
        return jpeg.tobytes()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        if self.proc:
            self.proc.terminate()
            self.proc.wait()

# rpicam-vid H264 스트림 명령어
cmd = [
    "rpicam-vid",
    "-t", "0",
    "-o", "-",  # stdout
    "--width", "640",
    "--height", "480",
    "--framerate", "30",
    "-f", "h264"  # H264 출력
]

camera = CameraH264(cmd=cmd)

def generate_frames():
    while True:
        frame_bytes = camera.get_frame_bytes()
        if frame_bytes is None:
            time.sleep(0.01)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "<h1>Raspberry Pi Camera Stream (H264)</h1><img src='/video_feed'>"

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    finally:
        camera.stop()
