import torch
import glm

def fov_to_focal_length(fov: float, width: int) -> float:
    """
    Convert field of view (FOV) to focal length.

    Args:
        fov (float): Field of view in degrees (horizontal).
        width (int): Image width in pixels.

    Returns:
        float: Focal length in pixels.
    """
    fov_tensor = torch.tensor(fov, dtype=torch.float32)
    focal_length = width / (2 * torch.tan(torch.deg2rad(fov_tensor / 2)))
    return focal_length

def pos_angle_to_tform_cam2world(camera_pos: glm.vec3, camera_angle: glm.vec3) -> torch.Tensor:
    """
    Create a transformation matrix from camera position and angles using glm.

    Args:
        camera_pos (glm.vec3): Camera position as glm.vec3 (x, y, z).
        camera_angle (glm.vec3): Camera angles as glm.vec3 (pitch, yaw, roll) in radians.

    Returns:
        torch.Tensor: 4x4 transformation matrix from camera to world.
    """
    # Extract pitch, yaw, roll (radians)
    pitch = camera_angle.x
    yaw = camera_angle.y
    roll = camera_angle.z

    # Create rotation matrices using glm
    rotation_x = glm.rotate(glm.mat4(1.0), pitch, glm.vec3(1.0, 0.0, 0.0))
    rotation_y = glm.rotate(glm.mat4(1.0), yaw, glm.vec3(0.0, 1.0, 0.0))
    rotation_z = glm.rotate(glm.mat4(1.0), roll, glm.vec3(0.0, 0.0, 1.0))

    # Combine rotations (note the order: roll -> yaw -> pitch)
    rotation_matrix = rotation_z * rotation_y * rotation_x

    # Create the transformation matrix
    tform_cam2world = glm.translate(glm.mat4(1.0), camera_pos) * rotation_matrix

    # Convert glm.mat4 to torch.Tensor
    # Use ctypes to convert the pointer to a list
    tform_cam2world_list = [tform_cam2world[i][j] for i in range(4) for j in range(4)]
    tform_cam2world_tensor = torch.tensor(tform_cam2world_list, dtype=torch.float32).reshape(4, 4).T

    return tform_cam2world_tensor


def meshgrid_xy(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mimick np.meshgrid(..., indexing="xy") in pytorch. torch.meshgrid only allows "ij" indexing.
    (If you're unsure what this means, safely skip trying to understand this, and run a tiny example!)

    Args:
      tensor1 (torch.Tensor): Tensor whose elements define the first dimension of the returned meshgrid.
      tensor2 (torch.Tensor): Tensor whose elements define the second dimension of the returned meshgrid.
    """
    # TESTED
    ii, jj = torch.meshgrid(tensor1, tensor2)
    return ii.transpose(-1, -2), jj.transpose(-1, -2)

def get_ray_bundle(height: int, width: int, focal_length: float, tform_cam2world: torch.Tensor):
  r"""Compute the bundle of rays passing through all pixels of an image (one ray per pixel).

  Args:
    height (int): Height of an image (number of pixels).
    width (int): Width of an image (number of pixels).
    focal_length (float or torch.Tensor): Focal length (number of pixels, i.e., calibrated intrinsics).
    tform_cam2world (torch.Tensor): A 6-DoF rigid-body transform (shape: :math:`(4, 4)`) that
      transforms a 3D point from the camera frame to the "world" frame for the current example.

  Returns:
    ray_origins (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the centers of
      each ray. `ray_origins[i][j]` denotes the origin of the ray passing through pixel at
      row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
    ray_directions (torch.Tensor): A tensor of shape :math:`(width, height, 3)` denoting the
      direction of each ray (a unit vector). `ray_directions[i][j]` denotes the direction of the ray
      passing through the pixel at row index `j` and column index `i`.
      (TODO: double check if explanation of row and col indices convention is right).
  """
  # TESTED
  ii, jj = meshgrid_xy(
      torch.arange(width).to(tform_cam2world),
      torch.arange(height).to(tform_cam2world)
  )
  directions = torch.stack([(ii - width * .5) / focal_length,
                            -(jj - height * .5) / focal_length,
                            -torch.ones_like(ii)
                           ], dim=-1)
  ray_directions = torch.sum(directions[..., None, :] * tform_cam2world[:3, :3], dim=-1)
  ray_origins = tform_cam2world[:3, -1].expand(ray_directions.shape)
  return ray_origins, ray_directions

def glm_mat4_to_torch(m: glm.mat4) -> torch.Tensor:
    """
    Convert a GLM mat4 (column-major) into a (4,4) PyTorch tensor (row-major)
    so that (TorchMatrix @ vector) matches (GLMmatrix * vector).
    """
    arr = torch.tensor(m.to_list(), dtype=torch.float32)  # column-major
    arr = arr.reshape(4, 4).t()                           # transpose => row-major for Torch
    return arr

def make_view_matrix(pos: glm.vec3, euler_rad: glm.vec3) -> glm.mat4:
    """Build a view matrix with Euler angles (pitch, yaw, roll) in degrees.
       We'll assume the order: X (pitch), Y (yaw), Z (roll).
       The camera looks 'down -Z' after these rotations if angles=0.
    """
    view = glm.mat4(1.0)

    # NOTE: Negative angles for the inverse of the camera transform
    # 1) rotate by -pitch about X
    view = glm.rotate(view, -euler_rad.x, glm.vec3(1,0,0))
    # 2) rotate by -yaw about Y
    view = glm.rotate(view, -euler_rad.y, glm.vec3(0,1,0))
    # 3) rotate by -roll about Z
    view = glm.rotate(view, -euler_rad.z, glm.vec3(0,0,1))

    # Then translate by -camera_position
    view = glm.translate(view, -pos)
    return view

def blender_to_opengl(vecs_blender: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of 3D coords [*, 3] from Blender's system (Z up, Y forward)
    to an OpenGL-like system (Y up, Z forward).

    The last dimension must be 3 (x, y, z). The result has the same shape, 
    just re-mapped per vector:
        x_o = x_b
        y_o = -z_b
        z_o = y_b
    """
    # Split out x_b, y_b, z_b
    x_b = vecs_blender[..., 0]
    y_b = vecs_blender[..., 1]
    z_b = vecs_blender[..., 2]

    # Apply the mapping
    x_o = x_b
    y_o = -z_b
    z_o = y_b

    # Re-stack into the same shape [*, 3]
    return torch.stack([x_o, y_o, z_o], dim=-1)
