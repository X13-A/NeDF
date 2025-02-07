import torch

def generate_rays_from_projection(position, yaw, pitch, roll, projection_matrix, device="cpu"):
    """
    Generates a 100x100 grid of rays based on a projection matrix and a given position & orientation.

    Args:
        position (torch.Tensor): (3,) Tensor representing the camera's position in world space.
        yaw (float): Rotation around the vertical axis (Y) in degrees.
        pitch (float): Rotation around the lateral axis (X) in degrees.
        roll (float): Rotation around the forward axis (Z) in degrees.
        projection_matrix (torch.Tensor): (4, 4) Projection matrix from the dataset.
        device (str): Device to place the tensor ("cpu" or "cuda").

    Returns:
        torch.Tensor: (10000, 3) Tensor containing rotated ray directions in world space.
    """

    # Ensure all inputs are Float32
    position = position.to(dtype=torch.float32, device=device)
    projection_matrix = projection_matrix.to(dtype=torch.float32, device=device)

    # Convert degrees to radians
    yaw = torch.tensor(yaw * (torch.pi / 180), dtype=torch.float32, device=device)
    pitch = torch.tensor(pitch * (torch.pi / 180), dtype=torch.float32, device=device)
    roll = torch.tensor(roll * (torch.pi / 180), dtype=torch.float32, device=device)

    # 1. Generate a 100x100 grid of normalized screen coordinates (-1 to 1)
    width, height = 100, 100
    i, j = torch.meshgrid(
        torch.linspace(-1, 1, width, dtype=torch.float32, device=device),
        torch.linspace(-1, 1, height, dtype=torch.float32, device=device),
        indexing="ij"
    )

    # 2. Convert to homogeneous coordinates (x, y, -1, 1)
    screen_coords = torch.stack([i, j, -torch.ones_like(i), torch.ones_like(i)], dim=-1)  # (100, 100, 4)

    # 3. Transform through inverse projection matrix to get world space directions
    inv_proj_matrix = torch.linalg.inv(projection_matrix)  # Already Float32 now
    world_coords = torch.matmul(inv_proj_matrix, screen_coords.reshape(-1, 4, 1)).squeeze(-1)  # (10000, 4)

    # 4. Convert homogeneous coordinates to 3D by normalizing (divide by w)
    world_directions = world_coords[:, :3] / world_coords[:, 3:4]  # (10000, 3)

    # 5. Create rotation matrices for yaw, pitch, and roll
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    cos_pitch, sin_pitch = torch.cos(pitch), torch.sin(pitch)
    cos_roll, sin_roll = torch.cos(roll), torch.sin(roll)

    # Yaw (Rotation around Y axis)
    rot_yaw = torch.tensor([
        [cos_yaw, 0, sin_yaw],
        [0, 1, 0],
        [-sin_yaw, 0, cos_yaw]
    ], dtype=torch.float32, device=device)

    # Pitch (Rotation around X axis)
    rot_pitch = torch.tensor([
        [1, 0, 0],
        [0, cos_pitch, -sin_pitch],
        [0, sin_pitch, cos_pitch]
    ], dtype=torch.float32, device=device)

    # Roll (Rotation around Z axis)
    rot_roll = torch.tensor([
        [cos_roll, -sin_roll, 0],
        [sin_roll, cos_roll, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)

    # Combine rotations (R = R_yaw * R_pitch * R_roll)
    rotation_matrix = rot_yaw @ rot_pitch @ rot_roll

    # 6. Rotate the rays
    rotated_rays = torch.matmul(world_directions, rotation_matrix.T)

    # 7. Normalize rays to unit vectors
    ray_directions = rotated_rays / torch.norm(rotated_rays, dim=-1, keepdim=True)

    return ray_directions  # (10000, 3)