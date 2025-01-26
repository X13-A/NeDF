import torch
import glm
import numpy as np

def glm_mat4_to_torch(m: glm.mat4) -> torch.Tensor:
    """
    Convert a GLM mat4 (column-major) into a (4,4) PyTorch tensor (row-major)
    so that (TorchMatrix @ vector) matches (GLMmatrix * vector).
    """
    arr = torch.tensor(m.to_list(), dtype=torch.float32)  # column-major
    arr = arr.reshape(4, 4).t()                           # transpose => row-major for Torch
    return arr

def make_view_matrix(position: glm.vec3, direction: glm.vec3) -> glm.mat4:
    target = position + direction
    up = glm.vec3(0, 1, 0)
    return glm.lookAt(position, target, up)


def blender_to_opengl(vecs_blender: torch.Tensor) -> torch.Tensor:
    x_b = vecs_blender[..., 0]
    y_b = vecs_blender[..., 1]
    z_b = vecs_blender[..., 2]

    x_o = x_b
    y_o = z_b
    z_o = -y_b

    return torch.stack([x_o, y_o, z_o], dim=-1)


def blender_to_opengl_euler(blender_euler):
    # TODO: FIll method
    # https://blenderartists.org/t/blender-rotation-to-opengl-rotation/336818
    pass