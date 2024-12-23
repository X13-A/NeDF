import numpy as np
import os
import re
import torch
import cv2
from settings import *
from utils import fov_to_focal_length, pos_angle_to_tform_cam2world, get_ray_bundle

import torch
import numpy as np
import glm

def load_dataset():
    # Load camera transforms
    cameras = np.load(CAMERAS_PATH)

    # Near and far clipping thresholds for depth values
    near = 0.01
    far = 500.0

    # Initialize dataset
    dataset = {}

    print("Loading depth maps, generating rays, and sorting by depth...")
    # Load depth maps
    for filename in os.listdir(DEPTHS_PATH):
        # Extract the index from the filename
        index = re.search(r'\d+', filename)
        if index:
            index_value = int(index.group())

            # Ignore invalid entries
            angle_norm = np.linalg.norm(cameras[BASE_CAMERA_ANGLE_ENTRY][index_value])
            if angle_norm < 1 - 1e-6:
                print(f"- Skipping index {index_value} due to invalid angle norm: {angle_norm}")
                continue

            # Load the depth map
            depth_map_path = os.path.join(DEPTHS_PATH, filename)
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

            # Resize the depth map
            new_width = int(depth_map.shape[1] * DEPTHMAP_SIZE_RESCALE)
            new_height = int(depth_map.shape[0] * DEPTHMAP_SIZE_RESCALE)
            resized_depth_map = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Scale down the depth (too high for the current model)
            resized_depth_map = resized_depth_map * SCENE_SCALE

            # Compute focal length from FOV
            focal_length = fov_to_focal_length(FOV, new_width)

            # Convert position and angle to transformation matrix
            camera_pos = cameras[BASE_CAMERA_LOCATION_ENTRY][index_value]
            camera_angle = cameras[BASE_CAMERA_ANGLE_ENTRY][index_value]
            
            camera_pos_torch = torch.tensor(camera_pos, dtype=torch.float32)
            camera_angle_torch = torch.tensor(camera_angle, dtype=torch.float32)
            
            camera_pos_glm = glm.vec3(camera_pos[0], camera_pos[1], camera_pos[2])
            camera_angle_glm = glm.vec3(camera_angle[0], camera_angle[1], camera_angle[2])
            
            tform_cam2world = pos_angle_to_tform_cam2world(camera_pos_glm, camera_angle_glm)

            # Generate ray bundle
            ray_origins, ray_directions = get_ray_bundle(new_height, new_width, focal_length, tform_cam2world)

            # Compute projection matrix
            glm_projection_matrix = glm.perspective(glm.radians(FOV), new_width / new_height, near, far)
            projection_matrix = torch.tensor(glm_projection_matrix.to_list(), dtype=torch.float32)

            # Store in the dataset
            dataset[index_value] = {
                RAYS_ENTRY: {
                    RAY_ORIGINS_ENTRY: ray_origins,
                    RAY_DIRECTIONS_ENTRY: ray_directions
                },
                DEPTH_MAP_ENTRY: resized_depth_map,
                CAMERA_POS_ENTRY: camera_pos_torch,
                CAMERA_ANGLE_ENTRY: camera_angle_torch,
                CAMERA_TRANSFORM_ENTRY: tform_cam2world,
                CAMERA_PROJECTION_ENTRY: projection_matrix
            }

    dataset_size = len(dataset)
    print(f"Successfully generated {dataset_size} entries!")
    return dataset