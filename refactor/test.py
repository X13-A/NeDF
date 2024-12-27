import torch
from estimate_distance import estimate_distance
from dataset import load_dataset

# Hit point data
hit_origin = torch.tensor([-0.0907, -1.4868, -1.0066], device='cuda:0')
hit_direction = torch.tensor([0.1197, -0.8359, 0.5513], device='cuda:0')
hit_distance = 0.8924758434295654

# Compute midpoint along the ray
point_world = hit_origin + hit_direction * (hit_distance / 2)
print(f"Midpoint (World Space): {point_world}")

# Load dataset
dataset = load_dataset()

# Estimate distance to geometry
estimated_distance = estimate_distance(point_world, dataset, device='cuda:0')
print(f"Estimated Signed Distance at {point_world}: {estimated_distance}")