import torch
from estimate_distance import estimate_distance, estimate_distances
from dataset import load_dataset
from settings import *
import matplotlib.pyplot as plt
import math

print("Running test 1...")

def estimate_distance_simple(dataset, index, point_world, device="cuda"):
    depth_map = dataset[index][DEPTH_MAP_ENTRY]
    camera_pos = dataset[index][CAMERA_POS_ENTRY].to(device)
    view_matrix = dataset[index][CAMERA_TRANSFORM_ENTRY].to(device)
    projection = dataset[index][CAMERA_PROJECTION_ENTRY].to(device)
    point_world = point_world.to(device)

    # Convert point to homogeneous coordinates
    point_world_h = torch.cat([point_world, torch.tensor([1.0], device=device)])
    point_world_h = point_world_h.to(device)

    # Transform to camera space
    print(f"- VIEW MATRIX: {view_matrix}")
    point_camera = view_matrix @ point_world_h
    print(f"- VIEW COORDS: {point_camera}")

    # Transform to clip space and normalize to NDC
    point_clip = projection @ point_camera
    ndc_coords = point_clip[:3] / point_clip[3]
    print(f"- NDC COORDS: {ndc_coords}")
    
    # Map NDC to depth map coordinates
    uv_coords = torch.tensor([(ndc_coords[0] * 0.5 + 0.5), (ndc_coords[1] * 0.5 + 0.5)])
    x = int(uv_coords[1] * depth_map.shape[1])  # Y-coordinate maps to row
    y = int(uv_coords[0] * depth_map.shape[0])  # X-coordinate maps to column
    
    print(f"- UV COORDS: {uv_coords}")
    print(f"- X, Y COORDS: ({x}, {y})")

    # print(f"Camera Space: {point_camera}")
    if point_camera[2] <= 0:  # Behind the camera
        print("- Behind the camera")
        return None
    
    # Check if the point is within the valid NDC range
    if not (-1 < ndc_coords[0] < 1 and -1 < ndc_coords[1] < 1):
        print("- Outside of NDC range")
        return None

    # Sample the depth map
    sampled_depth = depth_map[y, x]

    # Compute distance from the camera to the point
    camera_to_point_distance = torch.norm(camera_pos - point_world[:3])

    # Compute the signed distance to geometry
    signed_distance = sampled_depth - camera_to_point_distance
    print(f"- Signed Distance: {signed_distance}")
    return signed_distance

# Load dataset
dataset = load_dataset()
index = 0

depth_map = dataset[index][DEPTH_MAP_ENTRY]
y, x = int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)
# print(f"Sampled X, Y:{x, y}")
depth = depth_map[y, x]

rays = dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY]
ray = rays[y, x]

angle = dataset[index][CAMERA_ANGLE_ENTRY]
# print(f"Angle: {angle}")
# print(f"Ray generated: {dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY][y, x]}")

pitch = angle[0]
yaw = angle[1]
roll = angle[2]
true_ray = torch.tensor([
    math.cos(yaw) * math.cos(pitch),
    math.sin(pitch),
    math.sin(yaw) * math.cos(pitch)
])

# print(f"True ray: {true_ray}")

positions = dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY]
position = positions[y, x]

# print(f"Camera Position: {position}")
# print(f"Depth: {depth}")

# TODO: Find out why it is negated
ray = ray * torch.tensor([1, 1, 1])
hit_point = position + ray * depth
# print(f"Hit Point: {hit_point}")

point_world = position + ray * (depth)
# print(f"Point World: {point_world}")

# Estimate distance to geometry
# print("Estimating distance (simple)...")
# estimate_distance_simple(dataset, index, point_world)

dataset = {index: dataset[index]}
print("\nEstimating distance (multi-view)...")
estimate_distance(point_world, dataset, "cuda")

# # plot the depth map
# plt.imshow(depth_map)
# plt.show()

print("Finished test 1.\n")