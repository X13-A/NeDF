import torch
from estimate_distance import estimate_distances
from dataset import load_dataset
from settings import *
import matplotlib.pyplot as plt

device = 'cuda:0'

dataset = load_dataset()
index = 0

depth_map = dataset[index][DEPTH_MAP_ENTRY]
y, x = int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)

print()
print(f"X, Y: {x}, {y}")
depth = depth_map[y, x]
angle = dataset[index][CAMERA_ANGLE_ENTRY]

rays = dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY]
ray = rays[y, x]

positions = dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY]
position = positions[y, x]

print(f"Camera Position: {position}")
print(f"Ray direction: {ray}")

point_world = position + ray * (depth)
print(f"Depth: {depth}")
print(f"Point (world): {point_world}")

# point_world = torch.tensor([ -5.4098,  27.6815,  -6.0000])

# Define a set of test points in world space (torch.Tensor of shape [N, 3])
# TODO: Why is it not working with offset points ? Maybe check z value of NDC coords
offset = 0.01
test_points = torch.stack([
    point_world,
    point_world + torch.tensor([1.0, 0.0, 0.0]) * offset,
    point_world - torch.tensor([1.0, 0.0, 0.0]) * offset,
    point_world + torch.tensor([0.0, 1.0, 0.0]) * offset,
    point_world - torch.tensor([0.0, 1.0, 0.0]) * offset
], dim=0).to(device=device)
print(f"Test points (world):\n{test_points}")

# filter out all entries excpet 0 in dataset
# dataset = {index: dataset[index]}

print("\nEstimating distance (tensor)...")
estimate_distances(test_points, dataset, device)

# plot the depth map
plt.imshow(depth_map)
plt.show()
