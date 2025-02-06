import glm
import numpy as np
import torch
from settings import *
from dataset import load_dataset
from utils import *

def estimate_distances(points_world: torch.Tensor, dataset, device, verbose=False):
    """
    Estimates the signed distance for each 3D point by sampling depth maps from multiple cameras.

    Args:
        points_world (torch.Tensor): (N, 3) tensor of 3D points in world space.
        dataset (dict): Dictionary of camera parameters (transform & projection matrices).
        device (torch.device): Device for computation (CPU or CUDA).
        verbose (bool): If True, prints debugging info.

    Returns:
        torch.Tensor: (N,) tensor containing estimated signed distances (NaN for invisible points).
    """
    num_points = points_world.shape[0]
    dtype = torch.float32

    # Initialiser les distances à un grand nombre positif (sera remplacé par la distance la plus proche)
    distances = torch.full((num_points,), float('inf'), device=device, dtype=dtype)

    # Convertir les points en coordonnées homogènes (N, 4)
    ones = torch.ones((num_points, 1), device=device, dtype=dtype)
    points_world_h = torch.cat([points_world, ones], dim=1).to(device)

    for key, value in dataset.items():
        if verbose:
            print(f"\n### Processing Camera {key} ###")

        # Récupérer les matrices de caméra et la carte de profondeur
        depth_map = value[DEPTH_MAP_ENTRY].to(device, dtype=dtype)
        camera_transform = value[CAMERA_TRANSFORM_ENTRY].to(device, dtype=dtype)
        projection_matrix = value[CAMERA_PROJECTION_ENTRY].to(device, dtype=dtype)

        # Position de la caméra
        camera_pos = camera_transform[:3, 3]

        # Transformer les points en espace caméra
        camera_transform_inv = torch.linalg.inv(camera_transform)
        points_camera = torch.matmul(points_world_h, camera_transform_inv.T)

        # Filtrer les points devant la caméra (Z < 0)
        valid_mask = points_camera[:, 2] < 0
        if not valid_mask.any():
            continue

        valid_points_camera = points_camera[valid_mask]
        valid_points_world = points_world[valid_mask]
        valid_indices = torch.where(valid_mask)[0]

        # Projection en espace NDC
        points_ndc_hom = torch.matmul(valid_points_camera, projection_matrix.T)
        points_ndc = points_ndc_hom[:, :3] / points_ndc_hom[:, 3:4]

        # Masquer les points hors des limites NDC
        ndc_mask = (
            (points_ndc[:, 0] >= -1) & (points_ndc[:, 0] <= 1) &  # X dans le frustum
            (points_ndc[:, 1] >= -1) & (points_ndc[:, 1] <= 1) &  # Y dans le frustum
            (points_ndc[:, 2] >= 0) & (points_ndc[:, 2] <= 1)     # Z dans les plans near/far
        )

        valid_indices = valid_indices[ndc_mask]
        valid_points_ndc = points_ndc[ndc_mask]
        valid_points_world = valid_points_world[ndc_mask]

        if valid_indices.numel() == 0:
            continue

        # Conversion des coordonnées NDC en indices de la carte de profondeur
        uv_coords = torch.stack([
            (valid_points_ndc[:, 0] * 0.5 + 0.5) * depth_map.shape[1],  # X -> width
            (valid_points_ndc[:, 1] * 0.5 + 0.5) * depth_map.shape[0]   # Y -> height
        ], dim=1).long()

        # S'assurer que les indices sont dans les limites
        x = torch.clamp(uv_coords[:, 0], 0, depth_map.shape[1] - 1)
        y = torch.clamp(depth_map.shape[0] - 1 - uv_coords[:, 1], 0, depth_map.shape[0] - 1)
        sampled_depth = depth_map[y, x]

        # Calcul des distances caméra-points
        camera_distances = torch.norm(valid_points_world - camera_pos, dim=1)

        # Distance signée (profondeur - distance de la caméra)
        signed_distance = sampled_depth - camera_distances

        # Mettre à jour uniquement si la distance est plus proche de zéro
        update_mask = torch.abs(signed_distance) < torch.abs(distances[valid_indices])
        distances[valid_indices[update_mask]] = signed_distance[update_mask]

    # Remplacer les valeurs infinies par NaN (points invisibles)
    distances[distances == float('inf')] = float('nan')
    
    # Clamp to 0
    distances = torch.clamp(distances, min=0)
    return distances


def check_visibility(points_world: torch.Tensor, dataset, device, verbose=False):
    """
    Checks the visibility of a batch of 3D points using multiple cameras.

    Args:
        points_world (torch.Tensor): (N, 3) tensor of 3D points in world space.
        dataset (dict): Dictionary of camera parameters (transform & projection matrices).
        device (torch.device): Device for computation (CPU or CUDA).
        verbose (bool): If True, prints debugging info.

    Returns:
        torch.Tensor: (N,) boolean tensor indicating visibility (True = visible).
    """
    num_points = points_world.shape[0]
    dtype = torch.float32
    visibility = torch.zeros((num_points,), dtype=torch.bool, device=device)

    # Convert points to homogeneous coordinates (N, 4)
    ones = torch.ones((num_points, 1), device=device, dtype=dtype)
    points_world_h = torch.cat([points_world, ones], dim=1).to(device)  # (N, 4)

    for key, value in dataset.items():
        print(f"\n### Camera {key} ###")

        # Get camera matrices
        camera_transform = value[CAMERA_TRANSFORM_ENTRY].to(device, dtype=dtype)
        projection_matrix = value[CAMERA_PROJECTION_ENTRY].to(device, dtype=dtype)

        # Transform to camera space
        camera_transform_inv = torch.linalg.inv(camera_transform)
        points_camera = torch.matmul(points_world_h, camera_transform_inv.T)  # No need for .T

        if verbose:
            print(f"Points (Camera Space):\n{points_camera}")

        # Filter out points behind the camera
        valid_mask = points_camera[:, 2] < 0
        if verbose:
            print(f"Valid mask (points in front of camera): {valid_mask}")

        if not valid_mask.any():
            continue  # Skip this camera if no valid points

        valid_points_camera = points_camera[valid_mask]
        if verbose:
            print(f"Valid Points (Camera Space):\n{valid_points_camera}")

        # Project to NDC space
        points_ndc_hom = torch.matmul(valid_points_camera, projection_matrix.T)  # No need for .T

        # Perform perspective division
        points_ndc = points_ndc_hom[:, :3] / points_ndc_hom[:, 3:4]  # Shape: (N_valid, 3)

        if verbose:
            print(f"Points (NDC):\n{points_ndc}")

        # Check if points are within the valid NDC range
        ndc_mask = (
            (points_ndc[:, 0] >= -1) & (points_ndc[:, 0] <= 1) &  # X within frustum
            (points_ndc[:, 1] >= -1) & (points_ndc[:, 1] <= 1) &  # Y within frustum
            (points_ndc[:, 2] >= 0) & (points_ndc[:, 2] <= 1)    # Z within near/far planes
        )

        valid_indices = torch.where(valid_mask)[0][ndc_mask]
        visibility[valid_indices] = True

    return visibility
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