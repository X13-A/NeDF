import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init

# Define the MLP for Signed Distance Function (SDF)
class SDFModel(nn.Module):
    def __init__(self):
        super(SDFModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output: signed distance to closest surface
        )

    def forward(self, x):
        return self.layers(x)

# Custom weight initialization
def custom_sdf_init(layer):
    if isinstance(layer, nn.Linear):
        # Use uniform initialization within a small range to bias towards 0.1 to 0.2
        init.uniform_(layer.weight, a=-0.05, b=0.05)
        init.uniform_(layer.bias, a=0.1, b=0.2)

# Initialize the model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SDFModel().to(device)
model.apply(custom_sdf_init) 
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training Parameters
num_epochs = 10000
learning_rate = 0.001
step_size = 0.1  # Initial step size for ray marching
threshold = 0.05  # Threshold for SDF to consider as near surface

# Dummy depth map data (replace this with actual data)
camera_positions = torch.tensor([[0, 0, -5], [5, 0, -5], [-5, 0, -5]], dtype=torch.float32).to(device)
ray_directions = torch.tensor([[0, 0, 1], [-1, 0, 1], [1, 0, 1]], dtype=torch.float32).to(device)
reference_depths = torch.tensor([28.3, 37.1, 46.5], dtype=torch.float32).to(device)

# Training
for epoch in range(num_epochs):
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for cam_pos, ray_dir, ref_depth in zip(camera_positions, ray_directions, reference_depths):
        # Generate multiple points along the ray
        steps = torch.linspace(0, ref_depth * 1.5, steps=50, device=device)
        points = cam_pos + steps.unsqueeze(1) * ray_dir

        # Ensure `points` requires gradients for computing SDF gradient
        points.requires_grad_(True)

        # Predict SDF values for all points
        sdf_vals = model(points)

        # Calculate depth where SDF is close to zero
        # Add epsilon to prevent zero SDF causing infinite loop
        estimated_depth = steps[torch.argmin(torch.abs(sdf_vals + 1e-6))]

        # Compute depth loss
        depth_loss = loss_fn(estimated_depth, ref_depth)

        # Eikonal Loss (gradient regularization)
        sdf_grad = torch.autograd.grad(outputs=sdf_vals.sum(), inputs=points, create_graph=True, retain_graph=True)[0]
        eikonal_loss = ((sdf_grad.norm(2, dim=-1) - 1) ** 2).mean()

        # Total loss
        # Increase or decrease the weight of the Eikonal loss based on experimental results
        loss = depth_loss + 0.1 * eikonal_loss
        total_loss = total_loss + loss

    # Backpropagation
    total_loss.backward(retain_graph=True)
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {total_loss.item()}")

# Inference
print("\n=== Inference ===")
for cam_pos, ray_dir, ref_depth in zip(camera_positions, ray_directions, reference_depths):
    # Perform ray marching for inference
    t = 0.0
    point = cam_pos + t * ray_dir
    point = point.unsqueeze(0)

    while t < ref_depth:
        sdf_val = model(point).item()
        
        if abs(sdf_val) < threshold:
            break

        t += max(sdf_val, step_size)
        point = (cam_pos + t * ray_dir).unsqueeze(0)

    predicted_depth = t
    print(f"Reference Depth: {ref_depth}, Predicted Depth: {predicted_depth}, Difference: {abs(predicted_depth - ref_depth)}")
