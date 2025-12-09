import subprocess
import cv2
import numpy as np

def rpicam_vid_stream():
    # rpicam-vid 명령어 설정
    cmd = [
        "rpicam-vid",
        "-t", "5000",
        "-o", "-",       # stdout으로 영상 출력
        "--width", "640",
        "--height", "480",
        "--framerate", "30",
        "--liba-format", "jpeg"  # JPEG 포맷으로 출력
    ]

    # subprocess로 실행
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10**8)

    data = b""
    frame_count = 0
    while True:
        chunk = proc.stdout.read(1024)
        if not chunk:
            break
        data += chunk

        # JPEG 프레임 경계 찾기
        start = data.find(b'\xff\xd8')  # JPEG SOI
        end = data.find(b'\xff\xd9')    # JPEG EOI
        if start != -1 and end != -1:
            jpg = data[start:end+2]
            data = data[end+2:]

            # OpenCV로 디코딩
            frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                cv2.imshow("RPiCam VID Stream", frame)
                
                # 프레임 저장 예제
                cv2.imwrite(f"frame_{frame_count:04d}.jpg", frame)
                frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    proc.terminate()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    rpicam_vid_stream()
