import torch
from estimate_distance import estimate_distances
from dataset import load_dataset
from settings import *

device = 'cuda'

dataset = load_dataset()
index = 0

depth_map = dataset[index][DEPTH_MAP_ENTRY]
y, x = int(depth_map.shape[0] / 2), int(depth_map.shape[1] / 2)

print()
print(f"X, Y: {x}, {y}")
depth = depth_map[y, x]
angle = dataset[index][CAMERA_ANGLE_ENTRY].to(device)

rays = dataset[index][RAYS_ENTRY][RAY_DIRECTIONS_ENTRY]
ray = rays[y, x].to(device)

positions = dataset[index][RAYS_ENTRY][RAY_ORIGINS_ENTRY]
position = positions[y, x].to(device)
position += torch.tensor([0.01,0,0], device=device)

print(f"Camera Position: {position}")
print(f"Ray direction: {ray}")
print(f"Depth: {depth}")

test_points = torch.empty((0, 3), device=device)
n = 10

for i in range(0, n+1):
    pos = position + (ray * (depth / n * i + 0.001)).unsqueeze(0)
    test_points = torch.cat((test_points, pos), dim=0)


print(f"Test points (world):\n{test_points}")

# filter out all entries excpet 0 in dataset
dataset = {index: dataset[index]}

print("\nEstimating distance (tensor)...")
estimate_distances(test_points, dataset, device)