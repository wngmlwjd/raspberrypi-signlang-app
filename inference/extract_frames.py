import os
import cv2
from typing import List, Optional

from config.config import VIDEO_PATH, FRAMES_DIR

def extract_frames(video_path: str = VIDEO_PATH,
                   save_dir: Optional[str] = FRAMES_DIR,
                   prefix: str = "frame",
                   save_frames: bool = True) -> List:
    """
    영상 파일에서 프레임을 추출하여 리스트로 반환하고, 필요시 파일로 저장

    Parameters
    ----------
    video_path : str
        입력 영상(.mp4 등) 파일 경로
    save_dir : str, optional
        프레임을 저장할 폴더 경로 (save_frames=True일 때만 사용)
    prefix : str
        저장할 프레임 파일명의 접두사 (기본값: "frame")
    save_frames : bool
        True이면 프레임을 파일로 저장, False면 저장하지 않고 메모리로만 반환

    Returns
    -------
    frames : list of ndarray
        영상에서 추출한 프레임 이미지 리스트
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"❌ 영상 파일을 찾을 수 없음: {video_path}")

    if save_frames and save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"❌ 영상 파일을 열 수 없음: {video_path}")

    frames = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 영상 끝

        frames.append(frame.copy())

        if save_frames and save_dir is not None:
            filename = f"{prefix}_{frame_idx:04d}.jpg"
            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, frame)

        frame_idx += 1

    cap.release()

    if save_frames and save_dir is not None:
        print(f"✅ 총 {frame_idx}개의 프레임을 저장했습니다 → {save_dir}")
    else:
        print(f"✅ 총 {frame_idx}개의 프레임을 추출했습니다 (저장하지 않음)")

    return frames

# 사용 예시
# frames = extract_frames(save_frames=False)
# for f in frames:
#     # 바로 실시간 처리 가능
#     pass
