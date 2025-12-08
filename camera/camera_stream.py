# camera/camera_stream.py
import cv2
import threading

class CameraStream:
    def __init__(self, src=0, width=640, height=480):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.running = True

        # 스레드 시작
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def get_frame(self):
        if self.frame is None:
            return None

        # JPEG로 인코딩
        ret, jpeg = cv2.imencode(".jpg", self.frame)
        return jpeg.tobytes()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()
