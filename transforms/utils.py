import torch
import glm
import numpy as np
import math

def glm_mat4_to_torch(m: glm.mat4) -> torch.Tensor:
    """
    Convert a GLM mat4 (column-major) into a (4,4) PyTorch tensor (row-major)
    so that (TorchMatrix @ vector) matches (GLMmatrix * vector).
    """
    arr = torch.tensor(m.to_list(), dtype=torch.float32)  # column-major
    arr = arr.reshape(4, 4).t()                           # transpose => row-major for Torch
    return arr

def make_view_matrix(position: glm.vec3, yaw: float, pitch: float, roll: float) -> glm.mat4:
    """
    Create a 4x4 OpenGL view matrix from position and Blender Euler angles (yaw, pitch, roll in degrees).
    
    Args:
        position (glm.vec3): The position of the camera in world space.
        yaw (float): Rotation around the Y-axis (in degrees).
        pitch (float): Rotation around the X-axis (in degrees).
        roll (float): Rotation around the Z-axis (in degrees).
    
    Returns:
        glm.mat4: The resulting OpenGL view matrix.
    """
    # Convert angles from degrees to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    # Start with an identity matrix
    rotation_matrix = glm.mat4(1.0)

    # Apply rotations in Blender's convention (yaw → roll → pitch)
    rotation_matrix = glm.rotate(rotation_matrix, yaw_rad, glm.vec3(0, 1, 0))    # Yaw (Y-axis)
    rotation_matrix = glm.rotate(rotation_matrix, roll_rad, glm.vec3(1, 0, 0))   # Roll (X-axis)
    rotation_matrix = glm.rotate(rotation_matrix, -pitch_rad, glm.vec3(0, 0, 1)) # Pitch (Z-axis)

    # Create the translation matrix
    translation_matrix = glm.translate(glm.mat4(1.0), position)

    # Combine inverted rotation and translation
    view_matrix = rotation_matrix * translation_matrix

    return view_matrix

def create_transform_matrix_from_blender(yaw: float, pitch: float, roll: float, position: torch.Tensor):
    """
    Crée une matrice de transformation 4x4 dans l’espace de Blender en utilisant yaw, pitch et roll (en radians).
    La transformation est appliquée dans l’ordre Blender : Roll -> Pitch -> Yaw.
    """

    if not isinstance(position, torch.Tensor) or position.shape != (3,):
        raise ValueError("La position doit être un tensor PyTorch de forme (3,)")

    # Matrice identité
    transformation_matrix = glm.mat4(1.0)

    # Appliquer les rotations dans l’ordre Blender (Roll -> Pitch -> Yaw)
    transformation_matrix = glm.rotate(transformation_matrix, roll, glm.vec3(1, 0, 0))   # Roll autour de X
    transformation_matrix = glm.rotate(transformation_matrix, pitch, glm.vec3(0, 1, 0))  # Pitch autour de Y
    transformation_matrix = glm.rotate(transformation_matrix, yaw, glm.vec3(0, 0, 1))    # Yaw autour de Z

    # Appliquer la translation
    position_glm = glm.vec3(position[0].item(), position[1].item(), position[2].item())
    transformation_matrix[3] = glm.vec4(position_glm, 1.0)

    return transformation_matrix


def blender_to_opengl(vecs_blender: torch.Tensor) -> torch.Tensor:
    x_b = vecs_blender[..., 0]
    y_b = vecs_blender[..., 1]
    z_b = vecs_blender[..., 2]

    x_o = x_b
    y_o = z_b
    z_o = -y_b

    return torch.stack([x_o, y_o, z_o], dim=-1)


def blender_to_opengl_euler(blender_euler: torch.Tensor) -> torch.Tensor:
    # TODO: FIll method
    # Blender:
    # Yaw = Z
    # Pitch = X
    # Roll = Y
    # OpenGL:
    # Yaw = Y
    # Pitch = X
    # Roll = -Z

    yaw_b = blender_euler[..., 0]
    pitch_b = blender_euler[..., 1]
    roll_b = blender_euler[..., 2]

    yaw_o = roll_b
    pitch_o = pitch_b
    roll_o = -yaw_b
    return torch.stack([yaw_o, pitch_o, roll_o], dim=-1)