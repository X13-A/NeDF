import pygame
import torch
import glm
import numpy as np
from transform import Transform
from camera import Camera
from settings import *
from graphics import *
from window import *

screen = setup_window()
clock = pygame.time.Clock()

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# Initialize the camera
camera = Camera(position=glm.vec3(0.0, 0.0, 0.0), fov=CAMERA_FOV, aspect_ratio=CAMERA_ASPECT_RATIO, near=CAMERA_NEAR, far=CAMERA_FAR)

grid_u, grid_v = get_uvs()
rgb_image = to_rgb_255(grid_u, grid_v, np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.uint8))
uv_surface = pygame.surfarray.make_surface(rgb_image)

def loop():
    running = True
    while running:
        delta_time = clock.tick(60) / 1000.0  # 60 FPS target, `tick()` returns milliseconds
        
        # Event handling to quit the program
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.blit(uv_surface, (0, 0))

        camera.rotate(10 * delta_time, 0, 0)
        directions_world_space = get_directions(camera, device)
        directions_rgb = (directions_world_space * 255).clamp(0, 255).to(torch.uint8)
        directions_image = directions_rgb.cpu().numpy()
        directions_surface = pygame.surfarray.make_surface(directions_image)
        screen.blit(directions_surface, (0, 0))

        # Refresh the display
        pygame.display.flip()

loop()
pygame.quit()