import pyvista as pv
import numpy as np
from dataset import *
from utils import *
from estimate_distance import *
import math

# Load dataset
dataset = load_dataset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load arrow
arrow_mesh_blender = pv.read("models/Arrow0Yaw0Pitch0Roll_blender.obj")
arrow_mesh_gl = pv.read("models/Arrow0Yaw0Pitch0Roll_gl.obj")

# Create an interactive PyVista plotter
plotter = pv.Plotter()

def create_rotation_matrix_from_blender(yaw: float, pitch: float, roll: float):
    """
    Create a 4x4 rotation matrix using yaw, pitch, and roll (in degrees).
    Takes as input blender yaw pitch roll, and outputs the matching OpenGL rotation matrix
    """
    # Start with an identity matrix
    rotation_matrix = glm.mat4(1.0)
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    # Apply rotations in weird order
    rotation_matrix = glm.rotate(rotation_matrix, yaw_rad, glm.vec3(0, 1, 0))
    rotation_matrix = glm.rotate(rotation_matrix, roll_rad, glm.vec3(1, 0, 0))
    rotation_matrix = glm.rotate(rotation_matrix, -pitch_rad, glm.vec3(0, 0, 1))

    return rotation_matrix

transform_matrix = create_rotation_matrix_from_blender(yaw=30, pitch=45, roll=60)
revert_matrix = glm.inverse(transform_matrix)

arrow_mesh_gl.transform(np.array(transform_matrix))
# arrow_mesh_gl.transform(np.array(revert_matrix))

plotter.add_mesh(arrow_mesh_gl, color="lightblue", show_edges=True)

# Plot origin
origin = pv.Sphere(radius=0.25, center=np.array([0, 0 , 0]))
plotter.add_mesh(origin, color="blue")

# Add spheres and local coordinate system axes for cameras
for index, data in dataset.items():
    # Get camera position and direction in Blender's coordinate system
    camera_pos = data[CAMERA_POS_ENTRY] 
    camera_rays = data[RAY_DIRECTIONS_ENTRY].numpy()

    # Define the grid size for arrows
    grid_size = 10
    ray_height, ray_width = camera_rays.shape[0], camera_rays.shape[1]
    step_h = ray_height // (grid_size - 1)
    step_w = ray_width // (grid_size - 1)

    # Add a text label at the camera position
    plotter.add_point_labels([camera_pos.numpy()], [f"Camera {index}"], point_size=20, font_size=12, name=f"label_{index}")

    # Add the camera's grid of forward direction arrows
    forward_scale = 5.0  # Scale for the forward direction
    arrow_scale = 0.2  # Keep the arrow shafts thin
    for i in range(0, ray_height, step_h):
        for j in range(0, ray_width, step_w):
            ray_direction = camera_rays[i, j]  # Get the ray direction
            plotter.add_arrows(camera_pos, ray_direction, color="purple", mag=forward_scale)


# Show the plot with axes
plotter.show_axes()
plotter.show()

# TODO: Use angles as forward vectors don't preserve all rotations
