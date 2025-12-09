import subprocess
import time

counter = 0

while True:
    # 파일 이름 생성 (3자리 0패딩)
    filename = f"test_{counter:03d}.jpg"
    
    # rpicam-dpeg 명령어 실행
    subprocess.run(["rpicam-jpeg", "--output", filename, "--timeout", "500"])
    
    # 카운터 증가
    counter += 1
    
    # 반복 간격 (timeout이 500ms이므로 sleep은 선택 사항)
    time.sleep(0.1)
