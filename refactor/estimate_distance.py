import glm
import numpy as np
import torch
from settings import *
from dataset import load_dataset
from utils import *

def estimate_distance(point_world: torch.Tensor, dataset, device):
    closest_positive = float('inf')
    farthest_negative = float('-inf')
    
    for key, value in dataset.items():
        depth_map = value[DEPTH_MAP_ENTRY]
        camera_pos = value[CAMERA_POS_ENTRY].to(device)
        view_matrix = value[CAMERA_TRANSFORM_ENTRY].to(device)
        projection = value[CAMERA_PROJECTION_ENTRY].to(device)
        point_world = point_world.to(device)

        # Convert point to homogeneous coordinates
        point_world_h = torch.cat([point_world, torch.tensor([1.0], device=device)])
        
        # Transform to camera space
        point_camera = view_matrix @ point_world_h
        print(f"Camera Space: {point_camera}")
        if point_camera[2] <= 0:  # Behind the camera
            continue

        # Transform to clip space and normalize to NDC
        point_clip = projection @ point_camera
        ndc_coords = point_clip[:3] / point_clip[3]

        print(f"NDC Coords: {ndc_coords}")

        # Check if the point is within the valid NDC range
        if not (-1 <= ndc_coords[0] <= 1 and -1 <= ndc_coords[1] <= 1 and -1 <= ndc_coords[2] <= 1):
            continue

        # Map NDC to depth map coordinates
        uv_coords = torch.tensor([(ndc_coords[0] * 0.5 + 0.5), (ndc_coords[1] * 0.5 + 0.5)], device=device)
        x = int(uv_coords[1] * depth_map.shape[0])  # Y-coordinate maps to row
        y = int(uv_coords[0] * depth_map.shape[1])  # X-coordinate maps to column

        # Sample the depth map
        sampled_depth = depth_map[x, y]

        # Compute distance from the camera to the point
        camera_to_point_distance = torch.norm(camera_pos - point_world)

        # Compute the signed distance to geometry
        signed_distance = sampled_depth - camera_to_point_distance
        
        # Update positive and negative distances separately
        if signed_distance >= 0:
            closest_positive = min(closest_positive, signed_distance)
        else:
            farthest_negative = max(farthest_negative, signed_distance)
    
    # Decide the final distance
    if closest_positive != float('inf') and farthest_negative != float('-inf'):
        # Both positive and negative distances exist
        final_distance = closest_positive if abs(closest_positive) < abs(farthest_negative) else farthest_negative
    elif closest_positive != float('inf'):
        # Only positive distances exist
        final_distance = closest_positive
    elif farthest_negative != float('-inf'):
        # Only negative distances exist
        final_distance = farthest_negative
    else:
        # No valid distance found
        final_distance = None

    print(f"Estimated distance ({point_world}): {final_distance}")
    return final_distance