import pygame
from settings import *

def setup_window():
    # Initialize Pygame and create a window
    pygame.init()
    screen = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT))
    pygame.display.set_caption("NeRFs")

    # Set the window icon
    icon = pygame.image.load('icon.png')
    pygame.display.set_icon(icon)

    return screen
