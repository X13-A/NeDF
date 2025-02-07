import pyvista as pv
import numpy as np
from modules.dataset import *
from modules.utils import *
from modules.estimate_distance import *
import math
from modules.settings import *
from modules.visualize import *
from modules.checkpoint import * 
from modules.model import *

import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dataset = {0: dataset[0]}

# Create an interactive PyVista plotter
plotter = pv.Plotter()

# Load the robot mesh
robot_mesh = pv.read("models/Robot Rouen Axes.obj")
plotter.add_mesh(robot_mesh, color="lightblue", show_edges=True)

# Plot origin
origin = pv.Sphere(radius=0.25, center=np.array([0, 0 , 0]))
plotter.add_mesh(origin, color="blue")

# Add spheres and local coordinate system axes for cameras
for index, data in dataset.items():
    # Get camera position and direction in Blender's coordinate system
    camera_transform = data[CAMERA_TRANSFORM_ENTRY]
    camera_projection = data[CAMERA_PROJECTION_ENTRY]

    right_vector_from_transform = camera_transform[:3, 0]
    up_vector_from_transform = camera_transform[:3, 1]
    forward_vector_from_transform = camera_transform[:3, 2]
    camera_pos_from_transform = camera_transform[:3, 3]

    scale = 2.0  # Échelle des vecteurs
    plotter.add_arrows(camera_pos_from_transform.numpy(), right_vector_from_transform.numpy() * scale, color="red", mag=1, label="Right")
    plotter.add_arrows(camera_pos_from_transform.numpy(), up_vector_from_transform.numpy() * scale, color="green", mag=1, label="Up")
    plotter.add_arrows(camera_pos_from_transform.numpy(), forward_vector_from_transform.numpy() * scale, color="blue", mag=1, label="Forward")

    # Add a text label at the camera position
    plotter.add_point_labels([camera_pos_from_transform.numpy()], [f"Camera {index}"], point_size=20, font_size=12, name=f"label_{index}")

    # Compute frustum corners and edges
    frustum_corners, frustum_edges = create_frustum(camera_projection, camera_transform, 0.010)
    # Generate frustum as a line mesh
    frustum_lines = create_frustum_lines(frustum_corners, frustum_edges)
    # Add the frustum as a **wireframe** without removing existing objects
    plotter.add_mesh(frustum_lines, color="cyan", line_width=2, opacity=0.7)


lr = 1e-5

# Model and optimizer
model = SimpleUDFModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Load checkpoint
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
start_epoch, best_loss = load_checkpoint(model, optimizer, os.path.join(checkpoint_dir, CHECKPOINT_NAME))

plotter.add_key_event("space", lambda: render(plotter=plotter, dataset=dataset, model=model, device=device))
plotter.show_axes()
plotter.show()