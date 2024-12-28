import torch
from estimate_distance import estimate_distance
from dataset import load_dataset
from settings import *
import matplotlib.pyplot as plt
import math

# Hit point data
# hit_origin = torch.tensor([-0.0907, -1.4868, -1.0066], device='cuda:0')
# hit_direction = torch.tensor([0.1197, -0.8359, 0.5513], device='cuda:0')
# hit_distance = 0.8924758434295654

# # Compute midpoint along the ray
# point_world = hit_origin + hit_direction * (hit_distance / 2)
# print(f"Midpoint (World Space): {point_world}")

# Load dataset
dataset = load_dataset()
index = 1

depth_map = dataset[index][DEPTH_MAP_ENTRY]
y, x = int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)
print(f"Sampled X, Y:{x, y}")
depth = depth_map[y, x]

rays = dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY]
ray = rays[y, x]

angle = dataset[index][CAMERA_ANGLE_ENTRY]
print(f"Angle: {angle}")
print(f"Ray generated: {dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY][y, x]}")

pitch = angle[0]
yaw = angle[1]
roll = angle[2]
true_ray = torch.tensor([
    math.cos(yaw) * math.cos(pitch),
    math.sin(pitch),
    math.sin(yaw) * math.cos(pitch)
])

print(f"True ray: {true_ray}")

positions = dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY]
position = positions[y, x]

print(f"Camera Position: {position}")
print(f"Depth: {depth}")

# TODO: Find out why it is negated
ray = ray * torch.tensor([1, 1, 1])
hit_point = position + ray * depth
print(f"Hit Point: {hit_point}")

point_world = position + ray * (depth)
print(f"Point World: {point_world}")

def estimate_distance_simple(dataset, index, point_world):
    depth_map = dataset[index][DEPTH_MAP_ENTRY]
    camera_pos = dataset[index][CAMERA_POS_ENTRY]
    view_matrix = dataset[index][CAMERA_TRANSFORM_ENTRY]
    projection = dataset[index][CAMERA_PROJECTION_ENTRY]

    # Convert point to homogeneous coordinates
    point_world_h = torch.cat([point_world, torch.tensor([1.0])])

    # Transform to camera space
    point_camera = view_matrix @ point_world_h
    print(f"Point (camera space): {point_camera}")

    # Transform to clip space and normalize to NDC
    point_clip = projection @ point_camera
    ndc_coords = point_clip[:3] / point_clip[3]
    print(f"NDC Coords: {ndc_coords}")
    
    # Map NDC to depth map coordinates
    uv_coords = torch.tensor([(ndc_coords[0] * 0.5 + 0.5), (ndc_coords[1] * 0.5 + 0.5)])
    x = int(uv_coords[1] * depth_map.shape[1])  # Y-coordinate maps to row
    y = int(uv_coords[0] * depth_map.shape[0])  # X-coordinate maps to column
    
    print(f"UV Coords: {uv_coords}")
    print(f"X, Y Coords: ({x}, {y})")

    # print(f"Camera Space: {point_camera}")
    if point_camera[2] <= 0:  # Behind the camera
        print("Behind the camera")
        return None
    
    # Check if the point is within the valid NDC range
    if not (-1 < ndc_coords[0] < 1 and -1 < ndc_coords[1] < 1):
        print("Outside of NDC range")
        return None

    # Sample the depth map
    sampled_depth = depth_map[y, x]
    print(sampled_depth)

    # Compute distance from the camera to the point
    camera_to_point_distance = torch.norm(camera_pos - point_world[:3])

    # Compute the signed distance to geometry
    signed_distance = sampled_depth - camera_to_point_distance
    print(f"Signed Distance: {signed_distance}")
    return signed_distance

# Estimate distance to geometry
estimate_distance_simple(dataset, index, point_world)

# plot the depth map
plt.imshow(depth_map)
plt.show()