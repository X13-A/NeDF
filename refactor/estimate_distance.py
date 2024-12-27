import glm
import numpy as np
import torch
from settings import *
from dataset import load_dataset
from utils import *

def estimate_distance(point_world : torch.tensor, dataset, device):
    min_dist = float('inf')
    for key, value in dataset.items():
        depth_map = value[DEPTH_MAP_ENTRY]
        camera_pos = value[CAMERA_POS_ENTRY]
        view_matrix = value[CAMERA_TRANSFORM_ENTRY]
        projection = value[CAMERA_PROJECTION_ENTRY]

        # Transfer everything to device
        camera_pos = camera_pos.to(device)
        view_matrix = view_matrix.to(device)
        projection = projection.to(device)
        point_world = point_world.to(device)

        point_world_h = torch.cat([point_world, torch.tensor([1.0], device=device)])   # [x, y, z, 1]
        camera_space_point = view_matrix @ point_world_h
        clip_space_point = projection @ camera_space_point
        ndc_coords = clip_space_point[:3] / clip_space_point[3]

        # Ignore invalid points
        if (ndc_coords[0] < -1 or ndc_coords[0] > 1 or ndc_coords[1] < -1 or ndc_coords[1] > 1 or ndc_coords[2] < -1 or ndc_coords[2] > 1):
            continue
        
        # Sample depth map
        uv_coords = torch.tensor([ndc_coords[0] * 0.5 + 0.5, ndc_coords[1] * 0.5 + 0.5])
        print("UV coords:", uv_coords)
        x, y = int(uv_coords[0] * depth_map.shape[0]), int(uv_coords[0] * depth_map.shape[1])
        print("Sampled depth map coords:", x, y)
        sampled_depth = depth_map[x, y]
        if sampled_depth < min_dist:
            min_dist = sampled_depth
    
    print(f"Estimated distance ({point_world}):", min_dist)
    return min_dist

# dataset = load_dataset()
# estimate_distance(torch.tensor([0, 0, -1], dtype=torch.float32), dataset, "cuda")

# projection   = dataset[0][CAMERA_PROJECTION_ENTRY].clone()
# view_matrix  = dataset[0][CAMERA_TRANSFORM_ENTRY].clone()
# camera_pos   = dataset[0][CAMERA_POS_ENTRY].clone()
# camera_angle = dataset[0][CAMERA_ANGLE_ENTRY].clone()



# # 2) Extract pitch, yaw (roll if needed)
# pitch = camera_angle[0]
# yaw   = camera_angle[1]

# point_world = torch.tensor([0, 0, -1], dtype=torch.float32)
# print("New world point:", point_world)

# # 5) Transform from world space -> camera space -> clip space -> NDC
# #    (assuming 'view_matrix' is indeed world->camera).
# point_world_h = torch.cat([point_world, torch.tensor([1.0])])   # [x, y, z, 1]

# camera_space_point = view_matrix @ point_world_h
# clip_space_point   = projection @ camera_space_point

# ndc_coords = clip_space_point[:3] / clip_space_point[3]

# print("Camera position:", camera_pos)
# print("Camera space:", camera_space_point)
# print("Clip space:", clip_space_point)
# print("NDC:", ndc_coords)

# Hit direction: tensor([ 0.1197, -0.8359,  0.5513], device='cuda:0')
# Hit distance: 0.8924758434295654
# Hit origin: tensor([-0.0907, -1.4868, -1.0066], device='cuda:0')
# Using previous info, get the point mid ray

# 1) Compute your point in OpenGL coords
# point_world_opengl = torch.tensor([-0.0907, -1.4868, -1.0066]) \
#                    + torch.tensor([0.1197, -0.8359, 0.5513]) * 0.8924758434295654 / 2

# # 2) Convert it to Blender coords
# point_world_blender = opengl_to_blender(point_world_opengl)

# print("OpenGL coords:  ", point_world_opengl)
# print("Blender coords:", point_world_blender)
# estimate_distance(point_world_blender, dataset, "cuda")