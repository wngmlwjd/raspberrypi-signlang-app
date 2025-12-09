from flask import Flask
import threading
import time
from inference import video_saver  # save_video()가 있는 모듈

app = Flask(__name__)

def start_saving():
    """
    5초마다 새로운 스레드에서 save_video() 호출
    """
    counter = 0
    while True:
        # 녹화 실행을 별도 스레드에서 수행
        t = threading.Thread(target=video_saver.save_video, args=(counter,))
        t.start()
        
        counter += 1
        time.sleep(6)  # 5초 간격으로 녹화 시작

@app.route("/")
def index():
    return "<h1>Video saving is running every 5 seconds asynchronously...</h1>"

if __name__ == "__main__":
    t = threading.Thread(target=start_saving, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=5000, debug=True)
