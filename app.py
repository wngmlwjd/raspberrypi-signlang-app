import time
from flask import Flask, Response, render_template
from camera.camera_stream import CameraStream
from inference.video_saver import save_video_frame, get_new_video_writer
from config.config import CMD, VIDEO_DIR, VIDEO_SIZE, VIDEO_FPS, SAVE_INTERVAL

app = Flask(__name__)

# 카메라 스트림 시작
camera = CameraStream(cmd=CMD)

# 영상 저장 설정
video_writer = None
video_start_time = 0
video_idx = 0

def generate():
    global video_writer, video_start_time, video_idx
    while True:
        frame = camera.get_frame()
        if frame is not None:
            current_time = time.time()

            # 일정 시간마다 새 영상 파일 생성
            if video_writer is None or current_time - video_start_time >= SAVE_INTERVAL:
                if video_writer is not None:
                    video_writer.release()

                video_writer = get_new_video_writer(VIDEO_DIR, video_idx, VIDEO_SIZE, VIDEO_FPS)
                video_start_time = current_time
                video_idx += 1

            # 영상 저장
            save_video_frame(frame, video_writer, VIDEO_SIZE)

            # 웹 스트리밍
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
        if video_writer is not None:
            video_writer.release()
