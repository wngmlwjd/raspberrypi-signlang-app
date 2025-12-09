import subprocess
import cv2
import numpy as np

def rpicam_stream():
    # rpicam 명령어
    cmd = [
        "rpicam",
        "--output", "-",      # stdout으로 출력
        "--nopreview",        # 미리보기 창 X
        "--timeout", "0",     # 무제한 실행
        "--width", "640",     # 해상도 설정
        "--height", "480",
        "--framerate", "30"   # FPS
    ]

    # subprocess로 rpicam 실행
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    data = b""
    while True:
        # stdout에서 버퍼 읽기
        chunk = proc.stdout.read(1024)
        if not chunk:
            break
        data += chunk

        # JPEG 프레임 경계 찾기
        start = data.find(b'\xff\xd8')  # JPEG SOI
        end = data.find(b'\xff\xd9')    # JPEG EOI
        if start != -1 and end != -1:
            jpg = data[start:end+2]     # JPEG 데이터
            data = data[end+2:]         # 남은 데이터

            # OpenCV로 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("RPiCam Stream", frame)

            # 종료 키
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rpicam_stream()
