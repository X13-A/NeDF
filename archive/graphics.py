from settings import *
import numpy as np
import glm
import torch

def get_uvs():
    u = np.linspace(0, 1, GRID_WIDTH, dtype=np.float32)
    v = np.linspace(0, 1, GRID_HEIGHT, dtype=np.float32)
    return np.meshgrid(u, v, indexing='ij')

def to_rgb_255(r, g, b):
    rgb_image = np.zeros((GRID_WIDTH, GRID_HEIGHT, 3), dtype=np.uint8)
    rgb_image[:, :, 0] = (r * 255).astype(np.uint8)  # Red channel
    rgb_image[:, :, 1] = (g * 255).astype(np.uint8)  # Green channel
    rgb_image[:, :, 2] = (b * 255).astype(np.uint8)  # Blue channel
    return rgb_image

def get_directions(camera, device):
    # Convert field of view to radians and calculate half-height and half-width
    fov_y = glm.radians(CAMERA_FOV)
    half_height = torch.tan(torch.tensor(fov_y / 2, device=device)) * CAMERA_NEAR
    half_width = half_height * CAMERA_ASPECT_RATIO

    # Create a mesh grid for pixel coordinates in the normalized device coordinate (NDC) space
    x = torch.linspace(-half_width, half_width, GRID_WIDTH, device=device)
    y = torch.linspace(half_height, -half_height, GRID_HEIGHT, device=device)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')  # grid_x, grid_y will both have shape (GRID_WIDTH, GRID_HEIGHT)

    # Stack x, y, and constant z=-1 into a 3D vector for each pixel (shape: (GRID_WIDTH, GRID_HEIGHT, 3))
    directions_camera_space = torch.stack([grid_x, grid_y, -torch.ones_like(grid_x)], dim=-1)

    # Convert the camera view matrix from GLM to PyTorch tensor
    inv_view_matrix_glm = glm.inverse(camera.view_matrix)  # Take the inverse of the view matrix
    inv_view_matrix = torch.tensor(inv_view_matrix_glm.to_list(), dtype=torch.float32, device=device)

    # Add a homogeneous component (w=0) to the direction vectors
    directions_homo = torch.cat([directions_camera_space, torch.zeros((*directions_camera_space.shape[:2], 1), device=device)], dim=-1)

    # Transform directions to world space by applying the inverse view matrix
    directions_world_space = torch.einsum('ij,hwj->hwi', inv_view_matrix, directions_homo)[:, :, :3]

    # Normalize the directions in world space
    directions_world_space = torch.nn.functional.normalize(directions_world_space, dim=-1)

    return directions_world_space