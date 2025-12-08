import pygame
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

# 초기화
pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(640, 480))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    frame_surface = pygame.surfarray.make_surface(np.rot90(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()
    clock.tick(30)

    rawCapture.truncate(0)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            camera.close()
            pygame.quit()
            exit()
