import os
import cv2

from config.config import VIDEO_PATH, FRAMES_DIR

def extract_frames(video_path: str = VIDEO_PATH, save_dir: str = FRAMES_DIR, prefix: str = "frame"):
    """
    영상 파일에서 프레임을 한 장씩 저장하는 함수

    Parameters
    ----------
    video_path : str
        입력 영상(.mp4 등) 파일 경로
    save_dir : str
        프레임을 저장할 폴더 경로
    prefix : str
        저장할 프레임 파일명의 접두사 (기본값: "frame")

    Returns
    -------
    int
        저장된 총 프레임 수
    """

    # 입력 영상 확인
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 영상 파일을 찾을 수 없음: {video_path}")

    # 출력 폴더 생성
    os.makedirs(save_dir, exist_ok=True)

    # 영상 읽기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 영상 파일을 열 수 없음: {video_path}")

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        # 파일명: frame_0001.jpg 형태
        filename = f"{prefix}_{frame_idx:04d}.jpg"
        save_path = os.path.join(save_dir, filename)

        # 프레임 저장
        cv2.imwrite(save_path, frame)
        frame_idx += 1

    cap.release()

    print(f"✅ 총 {frame_idx}개의 프레임을 저장했습니다 → {save_dir}")
    return frame_idx

# if __name__ == "__main__":
#     extract_frames()