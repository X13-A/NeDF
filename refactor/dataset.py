import numpy as np
import re
import torch
from settings import *
from utils import blender_to_opengl, fov_to_focal_length, glm_mat4_to_torch, make_view_matrix, pos_angle_to_tform_cam2world, get_ray_bundle
import imageio.v3 as iio
import torch
import numpy as np
import glm

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2

def load_dataset():
    # Load camera transforms
    cameras = np.load(CAMERAS_PATH)

    # Near and far clipping thresholds for depth values
    near = 0.01
    far = 500.0

    # Initialize dataset
    dataset = {}

    print("Loading depth maps, generating rays, and filtering by depth...")
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
            depth_map = depth_map[:, :, 0] # Keep only R channel

            # Resize the depth map
            new_width = int(depth_map.shape[1] * DEPTHMAP_SIZE_RESCALE)
            new_height = int(depth_map.shape[0] * DEPTHMAP_SIZE_RESCALE)
            resized_depth_map = cv2.resize(depth_map, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Convert depth map to torch tensor
            resized_depth_map = torch.tensor(resized_depth_map, dtype=torch.float32)

            # Get FOV
            fov = cameras[BASE_CAMERA_FOV_ENTRY][index_value]
            
            # Compute focal length from FOV
            focal_length = fov_to_focal_length(fov, new_width)

            # Convert position and angle to transformation matrix
            camera_pos = cameras[BASE_CAMERA_LOCATION_ENTRY][index_value]
            camera_angle = cameras[BASE_CAMERA_ANGLE_ENTRY][index_value]
            
            camera_pos_torch = blender_to_opengl(torch.tensor(np.array([camera_pos])))[0]
            camera_angle_torch = blender_to_opengl(torch.tensor(np.array([camera_angle])))[0]
            
            # print(f"- Processing index {index_value}")
            # print(f"  - Blender Camera Position: {camera_pos}")
            # print(f"  - Blender Camera Angle: {camera_angle}")
            # print(f"  - OpenGL Camera Position: {camera_pos_torch}")
            # print(f"  - OpenGL Camera Angle: {camera_angle_torch}")

            camera_pos_glm = glm.vec3(camera_pos_torch[0], camera_pos_torch[1], camera_pos_torch[2])
            camera_angle_glm = glm.vec3(camera_angle_torch[0], camera_angle_torch[1], camera_angle_torch[2])
            view_matrix_glm = make_view_matrix(camera_pos_glm, camera_angle_glm)
            view_matrix_torch = glm_mat4_to_torch(view_matrix_glm)
            
            cam2world_glm = glm.inverse(view_matrix_glm)
            cam2world_torch = glm_mat4_to_torch(cam2world_glm)

            # Generate ray bundle
            ray_origins, ray_directions = get_ray_bundle(new_height, new_width, focal_length, cam2world_torch)

            # Compute projection matrix
            projection_matrix_glm = glm.perspective(glm.radians(FOV), new_width / new_height, near, far)
            projection_matrix_torch = glm_mat4_to_torch(projection_matrix_glm)
            
            # Store in the dataset
            dataset[index_value] = {
                RAYS_ENTRY: {
                    RAY_ORIGINS_ENTRY: ray_origins,
                    RAY_DIRECTIONS_ENTRY: ray_directions
                },
                DEPTH_MAP_ENTRY: resized_depth_map,
                CAMERA_POS_ENTRY: camera_pos_torch,
                CAMERA_ANGLE_ENTRY: camera_angle_torch,
                CAMERA_TRANSFORM_ENTRY: view_matrix_torch,
                CAMERA_PROJECTION_ENTRY: projection_matrix_torch
            }

    dataset_size = len(dataset)
    print(f"Successfully generated {dataset_size} entries!")
    return dataset