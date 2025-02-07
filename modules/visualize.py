from modules.settings import *
from modules.render import *
import numpy as np
import pyvista as pv
import math
import torch
import matplotlib.pyplot as plt

# Function to extract PyVista camera position and rotation
def get_pyvista_camera_parameters(plotter):
    """
    Extracts the camera position and rotation (yaw, pitch, roll) from PyVista.

    Args:
        plotter (pv.Plotter): PyVista plotter instance.

    Returns:
        position (np.array): Camera position (x, y, z).
        euler_angles (tuple): Yaw, Pitch, Roll in degrees.
    """
    camera = plotter.camera

    # Extract camera position
    position = np.array(camera.position)

    # Get camera vectors
    focal_point = np.array(camera.focal_point)
    up_vector = np.array(camera.up)

    # Compute forward vector (normalized)
    forward_vector = focal_point - position
    forward_vector /= np.linalg.norm(forward_vector)

    # Compute Yaw (rotation around Y axis)
    yaw = math.degrees(np.arctan2(forward_vector[0], -forward_vector[2]))

    # Compute Pitch (rotation around X axis)
    pitch = math.degrees(np.arcsin(forward_vector[1]))

    # Compute Roll (rotation around Z axis)
    right_vector = np.cross(forward_vector, up_vector)
    up_vector_corrected = np.cross(right_vector, forward_vector)
    roll = math.degrees(np.arctan2(np.dot(up_vector, right_vector), np.dot(up_vector, up_vector_corrected)))

    return position, (yaw, pitch, roll)

def create_frustum(projection_matrix, camera_transform, far_scale=5.0):
    """Generate a correctly scaled frustum mesh from a projection matrix without changing the origin."""
    
    # Define the normalized frustum in clip space
    frustum_corners = np.array([
        [-1, -1, -1, 1], [1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, -1, 1],  # Near plane
        [-1, -1, 1, 1], [1, -1, 1, 1], [1, 1, 1, 1], [-1, 1, 1, 1]  # Far plane
    ])

    # Convert to camera space (multiply by inverse projection matrix)
    inv_proj = np.linalg.inv(projection_matrix)
    frustum_corners_cam = (inv_proj @ frustum_corners.T).T
    frustum_corners_cam /= frustum_corners_cam[:, 3].reshape(-1, 1)  # Normalize

    # ✅ Scale only the far-plane points (indices 4-7) without affecting the near-plane
    for i in range(4, 8):
        frustum_corners_cam[i, :3] *= far_scale  

    # Transform to world space using the camera's transformation matrix
    frustum_corners_world = (camera_transform @ frustum_corners_cam.T).T[:, :3]

    # Define frustum edges
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Near plane
        (4, 5), (5, 6), (6, 7), (7, 4),  # Far plane
        (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
    ]

    return frustum_corners_world, edges

def create_frustum_lines(frustum_corners, frustum_edges):
    """Create a line mesh for the camera frustum without affecting other objects."""
    
    if isinstance(frustum_corners, torch.Tensor):
        frustum_corners = frustum_corners.cpu().numpy()
    
    lines = []
    for edge in frustum_edges:
        lines.extend([2, edge[0], edge[1]])  # "2" means a line segment

    lines = np.array(lines, dtype=np.int32)  # Ensure correct type for PyVista

    return pv.PolyData(frustum_corners, lines=lines)

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

def generate_random_points(num_points, bounds):
    return torch.FloatTensor(
        np.random.uniform(bounds[0], bounds[1], size=(num_points, 3))
    )

def render(dataset, plotter, device, model):
    position, (yaw, pitch, roll) = get_pyvista_camera_parameters(plotter)
    print(f"position: {position}, yaw, pitch, roll: {yaw}, {pitch}, {roll}")

    position = torch.tensor(position, device=device, dtype=torch.float32)
    projection_matrix = dataset[0][CAMERA_PROJECTION_ENTRY]

    rays = generate_rays_from_projection(position, 0, 0, 0, projection_matrix, device=device)
    depth_map = march_rays(model, dataset, rays, position, num_samples=50, threshold=0.01, max_distance=10.0, device=device)

    depth_map_reshaped = depth_map.reshape(100, 100).cpu().detach().numpy()

    # Plot depth map
    plt.figure(figsize=(6, 6))
    plt.imshow(depth_map_reshaped, cmap="grey", interpolation="nearest")
    plt.colorbar(label="Depth (Distance Traveled)")
    plt.title("Ray Marching Depth Map")
    plt.axis("off")
    plt.show()

# # Generate random points
# spread = 5
# bounds = (-spread, spread)
# points_world = generate_random_points(50, bounds).to(device)
# # points_world[:, 2] = -5
# # points_world = torch.tensor([[0, -5, 0]]).to(device)
# visibilities = check_visibility(points_world, dataset, device, False)
# distances = estimate_distances(points_world, dataset, device, True)
# print(distances)

# # Ajouter des sphères au plotter en fonction des distances
# for i, point in enumerate(points_world.cpu().numpy()):
#     visibility = visibilities[i]
#     distance = distances[i]
    
#     if not visibility: continue
#     if not distance:
#         print(f"Error with point {point}")
#         continue
#     distance = math.fabs(distance)
#     if distance > 50: continue
    
#     scale = distance
#     opacity = 0.25
#     color = "yellow"
#     sphere = pv.Sphere(radius=scale, center=point)
#     plotter.add_mesh(sphere, color=color, opacity=opacity)

# # Sphere to move around
# sphere_pos = [0.0, 0.0, 0.0]
# sphere_actor = plotter.add_mesh(pv.Sphere(radius=0.5, center=sphere_pos), color="red")
# move_speed = 0.5

# def move_sphere(direction):
#     global sphere_pos, sphere_actor

#     if direction == "up":  # Move forward (positive Y)
#         sphere_pos[1] += move_speed
#     elif direction == "down":  # Move backward (negative Y)
#         sphere_pos[1] -= move_speed
#     elif direction == "left":  # Move left (negative X)
#         sphere_pos[0] -= move_speed
#     elif direction == "right":  # Move right (positive X)
#         sphere_pos[0] += move_speed
#     elif direction == "up_z":  # Move up (positive Z)
#         sphere_pos[2] += move_speed
#     elif direction == "down_z":  # Move down (negative Z)
#         sphere_pos[2] -= move_speed

#     # Compute new distance
#     dist = estimate_distances(torch.tensor([sphere_pos], device=device), dataset, device, False)
#     distance = dist[0] if dist[0] is not None else 0.25  # Avoid invalid values
#     print(f"{direction}: {sphere_pos}, Distance: {distance}")
#     distance = math.fabs(distance)
#     if distance > 100: distance = 0.25

#     # Remove the old sphere and add a new one with updated scale and transparency
#     plotter.remove_actor(sphere_actor)

#     sphere = pv.Sphere(radius=distance, center=sphere_pos)  # Scale sphere based on distance
#     sphere_actor = plotter.add_mesh(sphere, color="yellow", opacity=0.5)  # Transparent sphere


#     plotter.render()  # Refresh scene

# # Attach the movement function to key press events
# plotter.add_key_event("Up", lambda: move_sphere("up"))
# plotter.add_key_event("Down", lambda: move_sphere("down"))
# plotter.add_key_event("Left", lambda: move_sphere("left"))
# plotter.add_key_event("Right", lambda: move_sphere("right"))
# plotter.add_key_event("space", lambda: move_sphere("up_z"))
# plotter.add_key_event("Shift_L", lambda: move_sphere("down_z"))
