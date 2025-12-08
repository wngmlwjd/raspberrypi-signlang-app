import pygame
import cv2
import numpy as np

# 초기화
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

# 카메라 열기
cap = cv2.VideoCapture(0)  # 또는 /dev/video10 등 카메라 장치 경로

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        continue

    # OpenCV BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # numpy → pygame surface
    frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    # 이벤트 처리 (종료)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    clock.tick(30)

cap.release()
pygame.quit()
