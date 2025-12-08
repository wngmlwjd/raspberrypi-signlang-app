import pygame
import numpy as np
import cv2

pygame.init()
screen = pygame.display.set_mode((640, 480))
clock = pygame.time.Clock()

# frame: np.ndarray(BGR)
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
frame_surface = pygame.surfarray.make_surface(np.rot90(frame))
screen.blit(frame_surface, (0, 0))
pygame.display.flip()
clock.tick(30)
