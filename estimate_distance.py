import glm
import numpy as np
import torch
from settings import *
from dataset import load_dataset
from utils import *

def estimate_distance(point : torch.tensor, dataset):
    # iterate over dict entries
    for key, value in dataset.items():
        depth_map = value[DEPTH_MAP_ENTRY]
        camera_pos = value[CAMERA_POS_ENTRY]
        camera_angle = value[CAMERA_ANGLE_ENTRY]
        camera_transform = value[CAMERA_TRANSFORM_ENTRY]
        camera_projection = value[CAMERA_PROJECTION_ENTRY]

        # convert point to homogeneous coordinates
        homogeneous_point = torch.cat((point, torch.tensor([1.0])))

        # apply camera transform to point
        transformed_pos = torch.matmul(camera_transform, homogeneous_point)
        clip_space_pos = torch.matmul(camera_projection, transformed_pos)
        ndc_pos = clip_space_pos / clip_space_pos[3]
        print(ndc_pos)

dataset = load_dataset()
point = dataset[0][CAMERA_POS_ENTRY] + dataset[0][CAMERA_ANGLE_ENTRY] * 5
cam_pos = dataset[0][CAMERA_POS_ENTRY]
cam_pos = glm.vec3(cam_pos[0], cam_pos[1], cam_pos[2])
cam_angle = dataset[0][CAMERA_ANGLE_ENTRY]
cam_angle = glm.vec3(cam_angle[0], cam_angle[1], cam_angle[2])

cam2world = pos_angle_to_tform_cam2world(cam_pos, cam_angle).T
print(cam2world)
transformed_point = torch.matmul(cam2world, torch.cat((point, torch.tensor([1.0]))))

print(point)
print(transformed_point / transformed_point[3])

estimate_distance(point, dataset)