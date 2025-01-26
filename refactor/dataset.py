import numpy as np
import re
import torch
from settings import *
from utils import blender_to_opengl, glm_mat4_to_torch, make_view_matrix
import torch
import numpy as np
import glm

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def load_dataset(verbose = False):
    # Load camera transforms
    cameras = np.load(CAMERAS_PATH)

    # Near and far clipping thresholds for depth values
    near = 0.01
    far = 500.0

    # Initialize dataset
    dataset = {}

    print("Loading depth maps and generating rays...")
    # Load depth maps
    for filename in os.listdir(DEPTHS_PATH):
        try:
            # Extract the index from the filename
            index = re.search(r'\d+', filename)
            if index:
                index_value = int(index.group())

                # Load the depth map
                depth_map_path = os.path.join(DEPTHS_PATH, filename)
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth_map = depth_map[:, :, 0] # Keep only R channel

                # Resize the depth map
                new_width = int(depth_map.shape[1] * DEPTHMAP_SIZE_RESCALE)
                new_height = int(depth_map.shape[0] * DEPTHMAP_SIZE_RESCALE)
                resized_depth_map = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert depth map to torch tensor
                resized_depth_map = torch.tensor(resized_depth_map, dtype=torch.float32)

                # Load and resize camera rays
                base_camera_rays = cameras[BASE_CAMERA_RAYS_ENTRY][index_value]
                resized_camera_rays = cv2.resize(base_camera_rays, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Convert position and angle to transformation matrix
                base_camera_pos = cameras[BASE_CAMERA_LOCATION_ENTRY][index_value]
                
                # TODO: Increase precision by using BASE_CAMERA_ANGLE instead
                base_camera_forward = base_camera_rays[base_camera_rays.shape[0] // 2, base_camera_rays.shape[1] // 2]
                camera_rays_torch = torch.tensor(resized_camera_rays)
                camera_pos_torch = torch.tensor(base_camera_pos)
                camera_forward_torch = torch.tensor(base_camera_forward)

                # Convert to OpenGL coordinate system
                camera_pos_torch = blender_to_opengl(camera_pos_torch)
                camera_rays_torch = blender_to_opengl(camera_rays_torch)
                camera_forward_torch = blender_to_opengl(camera_forward_torch)

                print(camera_forward_torch)
                camera_pos_glm = glm.vec3(camera_pos_torch[0], camera_pos_torch[1], camera_pos_torch[2])
                camera_forward_glm = glm.vec3(camera_forward_torch[0], camera_forward_torch[1], camera_forward_torch[2])

                # TODO: Use and test new view matrix
                view_matrix_glm = make_view_matrix(camera_pos_glm, camera_forward_glm)
                view_matrix_torch = glm_mat4_to_torch(view_matrix_glm)
                
                # Compute projection matrix
                # TODO: Export FOV from blender
                projection_matrix_glm = glm.perspective(glm.radians(FOV), new_width / new_height, near, far)
                projection_matrix_torch = glm_mat4_to_torch(projection_matrix_glm)
                
                # Store in the dataset
                dataset[index_value] = {
                    DEPTH_MAP_ENTRY: resized_depth_map,
                    RAY_DIRECTIONS_ENTRY: camera_rays_torch,
                    CAMERA_POS_ENTRY: camera_pos_torch,
                    CAMERA_FORWARD_ENTRY: camera_forward_torch,
                    CAMERA_TRANSFORM_ENTRY: view_matrix_torch,
                    CAMERA_PROJECTION_ENTRY: projection_matrix_torch
                }
                print("success")
        except:
            print("Error")

    dataset_size = len(dataset)
    print(f"Successfully generated {dataset_size} entries!")
    return dataset

# Pre-pass function to filter depth and corresponding ray data
def filter_depth_map_torch(depth_map, ray_origins, ray_directions):
    # Flatten depth map and corresponding rays
    depth_flat = depth_map.flatten()
    ray_origins_flat = ray_origins.view(-1, 3)
    ray_directions_flat = ray_directions.view(-1, 3)
    print(depth_flat.shape)
    print(ray_origins_flat.shape)
    print(ray_directions_flat.shape)

    # Validate shapes
    if depth_flat.shape[0] != ray_origins_flat.shape[0] or depth_flat.shape[1] != ray_directions_flat.shape[1]:
        raise ValueError(f"Shape mismatch: depth {depth_flat.shape[0]}, rays {ray_origins_flat.shape[0]} {ray_directions_flat.shape[0]}")

    # Find maximum depth
    far_thresh = torch.max(depth_flat)

    print(far_thresh)
    # Filter out max depth values
    epsilon = 1e-4
    valid_mask = depth_flat < far_thresh - epsilon
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    # Calculate and print the percentage of sorted-out values
    removed_percentage = 100 * (depth_flat.shape[0] - valid_indices.shape[0]) / depth_flat.shape[0]
    print(f"Percentage of filtered-out values: {removed_percentage:.2f}%")

    # Extract corresponding valid data
    filtered_depth = depth_flat[valid_indices]
    filtered_ray_origins = ray_origins_flat[valid_indices]
    filtered_ray_directions = ray_directions_flat[valid_indices]

    # Compute 2D indices
    height, width = depth_map.shape
    valid_2d_indices = torch.stack(torch.unravel_index(valid_indices, (height, width)), dim=1)

    return filtered_depth, filtered_ray_origins, filtered_ray_directions, valid_indices, valid_2d_indices

# Post-process dataset to filter depth maps
def post_process_dataset(dataset, device="cpu"):
    print("Post-processing dataset...")

    for index, entry in dataset.items():
        try:
            # Store old values (for reconstruction)
            dataset[index]["OLD" + DEPTH_MAP_ENTRY] = dataset[index][DEPTH_MAP_ENTRY].clone()
            dataset[index]["OLD" + RAYS_ENTRY] = {
                RAY_ORIGINS_ENTRY: dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY].clone(),
                RAY_DIRECTIONS_ENTRY: torch.tensor(dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY], device=device),
            }
            dataset[index]["OLD" + CAMERA_POS_ENTRY] = dataset[index][CAMERA_POS_ENTRY].clone()

            # Extract depth map and ray data
            depth_map = entry[DEPTH_MAP_ENTRY].to(device)
            ray_origins = entry[RAYS_ENTRY][RAY_ORIGINS_ENTRY].to(device)
            ray_directions = entry[RAYS_ENTRY][RAY_DIRECTIONS_ENTRY].to(device)

            # Apply sorting and filtering
            filtered_depth, sorted_ray_origins, sorted_ray_directions, valid_indices, valid_2D_indices = filter_depth_map_torch(
                depth_map, ray_origins, ray_directions
            )

            # Update dataset entry
            dataset[index][FILTERED_DEPTH_MAP_ENTRY] = filtered_depth
            dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY] = sorted_ray_origins
            dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY] = sorted_ray_directions
            dataset[index][VALID_INDICES_ENTRY] = valid_indices
            dataset[index][VALID_2D_INDICES_ENTRY] = valid_2D_indices
        except Exception as e:
            print(f"Error processing index {index}: {e}")

    print("Post-processing complete!")
    return dataset