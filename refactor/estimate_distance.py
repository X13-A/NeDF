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
        depth_map = value[DEPTH_MAP_ENTRY].to(device)
        camera_pos = value[CAMERA_POS_ENTRY].to(device)
        view_matrix = value[CAMERA_TRANSFORM_ENTRY].to(device)
        projection = value[CAMERA_PROJECTION_ENTRY].to(device)
        point_world = point_world.to(device)

        # Convert point to homogeneous coordinates
        point_world_h = torch.cat([point_world, torch.tensor([1.0], device=device)])
        
        # Transform to camera space
        point_camera = view_matrix @ point_world_h
        # print(f"Camera Space: {point_camera}")
        if point_camera[2] <= 0:  # Behind the camera
            continue

        # Transform to clip space and normalize to NDC
        point_clip = projection @ point_camera
        ndc_coords = point_clip[:3] / point_clip[3]

        print(f"NDC Coords: {ndc_coords}")

        # Check if the point is within the valid NDC range
        if not (-1 <= ndc_coords[0] <= 1 and -1 <= ndc_coords[1] <= 1 and -1 <= ndc_coords[2] <= 1 + 1e-3):
            continue

        # Map NDC to depth map coordinates
        uv_coords = torch.tensor([(ndc_coords[0] * 0.5 + 0.5), (ndc_coords[1] * 0.5 + 0.5)], device=device)
        x = int(uv_coords[1] * depth_map.shape[1])  # Y-coordinate maps to row
        y = int(uv_coords[0] * depth_map.shape[0])  # X-coordinate maps to column

        print(f"[{key}] UV COORDS:\n {uv_coords}")
        print(f"[{key}] X, Y COORDS:\n {x}, {y}")

        # Sample the depth map
        sampled_depth = depth_map[y, x]

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

    print(f"- Estimated distance: {final_distance}")
    return final_distance

def check_visibility(points_world: torch.Tensor, dataset, device, verbose=False):
    num_points = points_world.shape[0]
    dtype = torch.float32
    visibility = torch.zeros((num_points,), dtype=torch.bool, device=device)

    # Convert points to homogeneous coordinates
    points_world_h = torch.cat([points_world, torch.ones((num_points, 1), device=device, dtype=dtype)], dim=1)
    points_world_h = points_world_h.to(device)

    for key, value in dataset.items():
        print()
        print(f"### {key} ###")
        view_matrix = value[CAMERA_TRANSFORM_ENTRY].to(device, dtype=dtype)
        projection = value[CAMERA_PROJECTION_ENTRY].to(device, dtype=dtype)

        # Transform to camera space
        points_camera = torch.matmul(points_world_h, view_matrix.T)
        if verbose: print(f"Points (Camera Space): {points_camera}")

        # Filter points behind the camera
        valid_mask = points_camera[:, 2] < 0
        if verbose: print(f"Valid mask (1): {valid_mask}")
        if not valid_mask.any():
            continue

        valid_points_camera = points_camera[valid_mask]
        if verbose: print(f"Valid Points (Camera Space): {valid_points_camera}")

        # Transform to clip space
        points_clip = torch.matmul(valid_points_camera, projection.T)
        if verbose: print(f"Points (Clip Space): {points_clip}")

        # Normalize to NDC
        ndc_coords = points_clip[:, :3] / points_clip[:, 3:4]
        if verbose: print(f"Points (NDC): {ndc_coords}")

        # Check if points are within the valid NDC range
        ndc_mask = (
            (ndc_coords[:, 0] >= -1) & (ndc_coords[:, 0] <= 1) &
            (ndc_coords[:, 1] >= -1) & (ndc_coords[:, 1] <= 1)
        )

        valid_indices = torch.where(valid_mask)[0][ndc_mask]
        visibility[valid_indices] = True

    return visibility

def estimate_distances(points_world: torch.Tensor, dataset, device, verbose = False):
    num_points = points_world.shape[0]
    dtype = torch.float32  # Ensure consistent data types
    final_distances = torch.full((num_points,), float('nan'), device=device, dtype=dtype)
    
    # Convert points to homogeneous coordinates
    points_world_h = torch.cat([points_world, torch.ones((num_points, 1), device=device, dtype=dtype)], dim=1)
    points_world_h = points_world_h.to(device)

    closest_positives = torch.full((num_points,), float('inf'), device=device, dtype=dtype)
    farthest_negatives = torch.full((num_points,), float('-inf'), device=device, dtype=dtype)

    for key, value in dataset.items():
        depth_map = value[DEPTH_MAP_ENTRY].to(device, dtype=dtype)
        camera_pos = value[CAMERA_POS_ENTRY].to(device, dtype=dtype)
        view_matrix = value[CAMERA_TRANSFORM_ENTRY].to(device, dtype=dtype)
        projection = value[CAMERA_PROJECTION_ENTRY].to(device, dtype=dtype)

        # Transform to camera space
        points_camera = torch.matmul(points_world_h, view_matrix.T)
        
        # Filter points behind the camera
        valid_mask = points_camera[:, 2] > 0
        if not valid_mask.any():
            continue

        valid_points_camera = points_camera[valid_mask]
        if verbose: print(f"Valid Points (Camera Space): {valid_points_camera}")

        # Transform to clip space
        points_clip = torch.mm(valid_points_camera, projection.T)
        if verbose: print(f"Points (Clip Space): {points_clip}")

        # Normalize to NDC
        ndc_coords = points_clip[:, :3] / points_clip[:, 3:4]
        if verbose: print(f"Points (NDC): {ndc_coords}")

        # Check if points are within the valid NDC range
        ndc_mask = (
            (ndc_coords[:, 0] >= -1) & (ndc_coords[:, 0] <= 1) &
            (ndc_coords[:, 1] >= -1) & (ndc_coords[:, 1] <= 1)
        )
        if not ndc_mask.any():
            continue

        valid_ndc_coords = ndc_coords[ndc_mask]
        valid_indices = torch.where(valid_mask)[0][ndc_mask]

        # Map NDC to depth map coordinates
        uv_coords = torch.stack([
            (valid_ndc_coords[:, 0] * 0.5 + 0.5),
            (valid_ndc_coords[:, 1] * 0.5 + 0.5)
        ], dim=1)
        x = (uv_coords[:, 1] * depth_map.shape[1]).long()
        y = (uv_coords[:, 0] * depth_map.shape[0]).long()
        if verbose: print(f"UV COORDS:\n {uv_coords}")
        if verbose: print(f"X, Y COORDS:\n {x}, {y}")

        # Sample depths for all points
        sampled_depths = depth_map[y, x]
        if verbose: print(f"Sampled Depths: {sampled_depths}")

        # Compute distance from the camera to the points
        camera_to_points_distance = torch.norm(camera_pos - points_world[valid_indices, :], dim=1)

        # Compute signed distances
        signed_distances = sampled_depths - camera_to_points_distance

        # Update positive and negative distances separately
        closest_positives[valid_indices] = torch.min(closest_positives[valid_indices], signed_distances)
        farthest_negatives[valid_indices] = torch.max(farthest_negatives[valid_indices], signed_distances)

    # Decide the final distances for all points
    for i in range(num_points):
        if closest_positives[i] != float('inf') and farthest_negatives[i] != float('-inf'):
            # Both positive and negative distances exist
            final_distances[i] = closest_positives[i] if abs(closest_positives[i]) < abs(farthest_negatives[i]) else farthest_negatives[i]
        elif closest_positives[i] != float('inf'):
            # Only positive distances exist
            final_distances[i] = closest_positives[i]
        elif farthest_negatives[i] != float('-inf'):
            # Only negative distances exist
            final_distances[i] = farthest_negatives[i]

    print(f"Estimated distances for points: {final_distances}")
    return final_distances