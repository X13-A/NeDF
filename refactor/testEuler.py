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

# camera_pos = 
# camera_angle = 
# transform_matrix = create_rotation_matrix_from_blender(yaw=30, pitch=45, roll=60, position=torch.tensor[])

# plotter.add_mesh(arrow_mesh_gl, color="lightblue", show_edges=True)

# Plot origin
origin = pv.Sphere(radius=0.25, center=np.array([0, 0 , 0]))
plotter.add_mesh(origin, color="blue")

# Add spheres and local coordinate system axes for cameras
for index, data in dataset.items():
    # Get camera position and direction in Blender's coordinate system
    camera_pos = data[CAMERA_POS_ENTRY] 
    camera_angle = data[CAMERA_ANGLE_BLENDER_ENTRY]
    
    print(f"Camera {index}:")
    print(camera_angle)
    transform = create_transform_matrix_from_blender(yaw=camera_angle[0], pitch=camera_angle[1], roll=camera_angle[2], position=camera_pos)
    transform = glm_mat4_to_torch(transform)
    right_vector_from_transform = transform[:3, 0]
    up_vector_from_transform = transform[:3, 1]
    forward_vector_from_transform = transform[:3, 2]
    camera_pos_from_transform = transform[:3, 3]

    # Ajouter des flèches pour chaque vecteur
    scale = 2.0  # Échelle des vecteurs

    plotter.add_arrows(camera_pos_from_transform.numpy(), right_vector_from_transform.numpy() * scale, color="red", mag=1, label="Right")
    plotter.add_arrows(camera_pos_from_transform.numpy(), up_vector_from_transform.numpy() * scale, color="green", mag=1, label="Up")
    plotter.add_arrows(camera_pos_from_transform.numpy(), forward_vector_from_transform.numpy() * scale, color="blue", mag=1, label="Forward")

    # TODO: Apply transform to camera_rays_blender and see if they align with camera_rays
    camera_rays_blender = data[RAY_DIRECTIONS_BLENDER_ENTRY].numpy()
    camera_rays = data[RAY_DIRECTIONS_ENTRY].numpy()
    print(transform)
    print()


    # Define the grid size for arrows
    grid_size = 10
    ray_height, ray_width = camera_rays.shape[0], camera_rays.shape[1]
    step_h = ray_height // (grid_size - 1)
    step_w = ray_width // (grid_size - 1)

    # Add a text label at the camera position
    plotter.add_point_labels([camera_pos.numpy()], [f"Camera {index}"], point_size=20, font_size=12, name=f"label_{index}")
    # plotter.add_point_labels([camera_pos_from_transform.numpy()], [f"Camera {index} (transform)"], point_size=20, font_size=12, name=f"label_{index}")

    # Add the camera's grid of forward direction arrows
    # forward_scale = 5.0  # Scale for the forward direction
    # arrow_scale = 0.2  # Keep the arrow shafts thin
    # for i in range(0, ray_height, step_h):
    #     for j in range(0, ray_width, step_w):
    #         ray_direction = camera_rays[i, j]  # Get the ray direction
    #         plotter.add_arrows(camera_pos, ray_direction, color="purple", mag=forward_scale)


# Show the plot with axes
plotter.show_axes()
plotter.show()

# TODO: Continue debugging transform matrix
# TODO: Maybe check yaw pitch roll and test it by hand
