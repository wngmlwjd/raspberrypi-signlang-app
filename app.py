from flask import Flask
import threading
import time
from inference import video_saver  # save_video()가 있는 모듈

app = Flask(__name__)

def start_saving():
    """
    save_video()를 5초마다 반복 실행
    """
    counter = 0
    
    while True:
        video_saver.save_video(counter)
        
        counter += 1

@app.route("/")
def index():
    return "<h1>Video saving is running every 5 seconds...</h1>"

if __name__ == "__main__":
    # 영상 저장 반복을 별도 스레드에서 시작
    t = threading.Thread(target=start_saving, daemon=True)
    t.start()

    # Flask 서버 실행
    app.run(host="0.0.0.0", port=5000, debug=True)
