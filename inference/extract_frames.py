import cv2
import os
from config import config

def extract_frames(video_path=None, output_dir="frames", frame_rate=1):
    """
    영상에서 프레임 추출
    """
    if video_path is None:
        video_path = config.VIDEO_PATH

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일이 없습니다: {video_path}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate) if fps > 0 else 1

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"총 {saved_count}개의 프레임을 {output_dir}에 저장했습니다.")
    return saved_count
