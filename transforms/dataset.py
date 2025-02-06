import numpy as np
import re
import torch
from settings import *
from utils import *
import torch
import numpy as np
import glm

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def load_dataset(verbose = False):
    # Load camera transforms
    cameras = np.load(CAMERAS_PATH)

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
                print(f"Processing camera {index_value}...")

                # Load the depth map
                depth_map_path = os.path.join(DEPTHS_PATH, filename)
                depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                depth_map = depth_map[:, :, 0] # Keep only R channel

                # Resize the depth map
                new_width = int(depth_map.shape[1] * DEPTHMAP_SIZE_RESCALE)
                new_height = int(depth_map.shape[0] * DEPTHMAP_SIZE_RESCALE)
                resized_depth_map = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
                resized_depth_map = torch.tensor(resized_depth_map, dtype=torch.float32)

                # Load and resize camera rays
                base_camera_rays = cameras[BASE_CAMERA_RAYS_ENTRY][index_value]
                resized_camera_rays = cv2.resize(base_camera_rays, (new_width, new_height), interpolation=cv2.INTER_AREA)
                camera_rays_torch = torch.tensor(resized_camera_rays)

                # Load view matrix
                view_matrix = cameras[BASE_CAMERA_TRANSFORM_ENTRY][index_value]
                view_matrix_torch = torch.tensor(view_matrix)

                # Load projection matrix
                projection_matrix = cameras[BASE_CAMERA_PROJECTION_ENTRY][index_value]
                projection_matrix_torch = torch.tensor(projection_matrix)

                # Store in the dataset
                dataset[index_value] = {
                    DEPTH_MAP_ENTRY: resized_depth_map,
                    RAY_DIRECTIONS_ENTRY: camera_rays_torch,
                    CAMERA_TRANSFORM_ENTRY: view_matrix_torch,
                    CAMERA_PROJECTION_ENTRY: projection_matrix_torch
                }
        except Exception as e:
            print(f"Error: {e}")

    dataset_size = len(dataset)
    print(f"Successfully generated {dataset_size} entries!")
    return dataset

# Pre-pass function to filter depth and corresponding ray data
def filter_depth_map_torch(depth_map, ray_directions):
    # Flatten depth map and corresponding rays
    depth_flat = depth_map.flatten()
    ray_directions_flat = ray_directions.view(-1, 3)

    # Find maximum depth
    far_thresh = torch.max(depth_flat)

    # Filter out max depth values
    epsilon = 1e-4
    valid_mask = depth_flat < far_thresh - epsilon
    valid_indices = torch.nonzero(valid_mask, as_tuple=True)[0]

    # Calculate and print the percentage of sorted-out values
    removed_percentage = 100 * (depth_flat.shape[0] - valid_indices.shape[0]) / depth_flat.shape[0]
    print(f"Percentage of filtered-out values: {removed_percentage:.2f}%")

    # Extract corresponding valid data
    filtered_depth = depth_flat[valid_indices]
    filtered_ray_directions = ray_directions_flat[valid_indices]

    # Compute 2D indices, not used for now
    height, width = depth_map.shape
    valid_2d_indices = torch.stack(torch.unravel_index(valid_indices, (height, width)), dim=1)

    return filtered_depth, filtered_ray_directions, valid_indices, valid_2d_indices

# Post-process dataset to filter depth maps
def post_process_dataset(dataset, device="cpu"):
    print("Post-processing dataset...")

    for index, entry in dataset.items():
        try:
            # Extract depth map and ray data
            depth_map = entry[DEPTH_MAP_ENTRY].to(device)
            ray_directions = entry[RAY_DIRECTIONS_ENTRY].to(device)

            # Apply sorting and filtering
            filtered_depth, filtered_ray_directions, valid_indices, valid_2D_indices = filter_depth_map_torch(depth_map, ray_directions)

            # Update dataset entry
            dataset[index][FILTERED_DEPTH_MAP_ENTRY] = filtered_depth
            dataset[index][FILTERED_RAY_DIRECTIONS_ENTRY] = filtered_ray_directions
        except Exception as e:
            print(f"Error processing index {index}: {e}")

    print("Post-processing complete!")
    return dataset