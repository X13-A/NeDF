import torch
from estimate_distance import estimate_distances
from dataset import load_dataset
from settings import *
import matplotlib.pyplot as plt
import math

print("Running test 2...")

dataset = load_dataset()
index = 0

depth_map = dataset[index][DEPTH_MAP_ENTRY]
y, x = int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)

depth = depth_map[y, x]
angle = dataset[index][CAMERA_ANGLE_ENTRY]

rays = dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY]
ray = rays[y, x]

positions = dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY]
position = positions[y, x]

ray = ray * torch.tensor([1, 1, 1])
point_world = position + ray * (depth)
# point_world = torch.tensor([ -5.4098,  27.6815,  -6.0000])

# Define a set of test points in world space (torch.Tensor of shape [N, 3])
test_points = torch.stack([
    point_world,
    point_world + torch.tensor([20.0, 0.0, 0.0]),
    point_world - torch.tensor([20.0, 0.0, 0.0]),
    point_world + torch.tensor([0.0, 20.0, 0.0]),
    point_world - torch.tensor([0.0, 20.0, 0.0])
], dim=0).to(device='cuda:0')

# Estimate distances for the test points
device = 'cuda:0'

# filter out all entries excpet 0 in dataset
# dataset = {index: dataset[index]}

print("\nEstimating distance (tensor)...")
estimate_distances(test_points, dataset, device)

print("Finished test 2.\n\n")
