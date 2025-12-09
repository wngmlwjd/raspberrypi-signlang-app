import cv2
import threading
import subprocess
import numpy as np

class CameraStream:
    def __init__(self, cmd=None):
        """
        cmd : list
            rpicam-vid 실행 명령어.
            예: ["rpicam-vid", "-t", "0", "-o", "-", "--width", "640", "--height", "480", "--framerate", "30"]
        """
        if cmd is None:
            raise ValueError("rpicam-vid 명령어를 cmd 인자로 전달해야 합니다.")

        self.cmd = cmd
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, bufsize=10**8
        )

        self.frame = None
        self.running = True
        self.buffer = b""

        # 카메라 스트림 업데이트용 스레드 시작
        self.thread = threading.Thread(target=self._update_stream, daemon=True)
        self.thread.start()

    def _update_stream(self):
        while self.running:
            try:
                chunk = self.proc.stdout.read(1024)
                if not chunk:
                    break
                self.buffer += chunk

                # JPEG 프레임 단위 추출
                start = self.buffer.find(b'\xff\xd8')
                end = self.buffer.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg = self.buffer[start:end + 2]
                    self.buffer = self.buffer[end + 2:]

                    frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        self.frame = frame

            except Exception as e:
                print("⚠️ Camera stream error:", e)
                break

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
