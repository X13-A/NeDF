import pyvista as pv
import numpy as np
from dataset import *
from utils import *
from estimate_distance import *
# Load dataset
dataset = load_dataset()
dataset = {1: dataset[1]}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the robot mesh
robot_mesh = pv.read("models/Robot Rouen Axes.obj")

# Create an interactive PyVista plotter
plotter = pv.Plotter()

plotter.add_mesh(robot_mesh, color="lightblue", show_edges=True)

def generate_random_points(num_points, bounds):
    return torch.FloatTensor(
        np.random.uniform(bounds[0], bounds[1], size=(num_points, 3))
    )

# Generate random points
spread = 20
bounds = (-spread, spread)
points_world = generate_random_points(500, bounds).to(device)
points_world[:, 1] = -5
points_world = torch.tensor([[0, 5, 0]]).to(device)

visibilities = check_visibility(points_world, dataset, device, True)

# Ajouter des sphères au plotter en fonction des distances
for i, point in enumerate(points_world.cpu().numpy()):
    visibility = visibilities[i].item()
    color = "blue" if visibility else "red"
    sphere = pv.Sphere(radius=0.1, center=point)
    plotter.add_mesh(sphere, color=color)

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

    # Add the camera position as a sphere
    sphere = pv.Sphere(radius=0.1, center=camera_pos)
    plotter.add_mesh(sphere, color="red")

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
