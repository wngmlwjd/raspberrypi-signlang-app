# test_camera_stream.py
from camera.camera_stream import CameraStream
import cv2
from config.config import CMD  # rpicam-vid 명령어

def main():
    # CameraStream 생성
    cam = CameraStream(cmd=CMD)
    print("✅ Camera stream started.")

    try:
        while True:
            frame_bytes = cam.get_frame()
            if frame_bytes is None:
                continue  # 아직 프레임이 준비되지 않음

            # JPEG → OpenCV 이미지
            frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 화면에 표시
            cv2.imshow("CameraStream Test", frame)

            # q 누르면 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()
        print("✅ Camera stream stopped.")

if __name__ == "__main__":
    main()
