DATA_PATH = "./data/alex"
CAMERAS_PATH = f"{DATA_PATH}/cameras_attributes.npz"
DEPTHS_PATH = f"{DATA_PATH}/depth"
RAYS_PATH = f"{DATA_PATH}/rays"

FOV = 60
DEPTHMAP_SIZE_RESCALE = 100/1920
TARGET_IMAGE_SIZE = [100, 100]

# BASE DATASET STRUCTURE
BASE_CAMERA_RAYS_ENTRY = "ray_maps"
BASE_CAMERA_TRANSFORM_ENTRY = "cameras_matrices"
BASE_CAMERA_PROJECTION_ENTRY = "projection_matrices"

# TARGET DATASET STRUCTURE
RAY_DIRECTIONS_ENTRY = "ray_directions"
DEPTH_MAP_ENTRY = "depth_map"
FILTERED_DEPTH_MAP_ENTRY = "filtered_depth_map"
VALID_INDICES_ENTRY = "valid_indices"
VALID_2D_INDICES_ENTRY = "valid_2D_indices"
CAMERA_TRANSFORM_ENTRY = "camera_transform"
CAMERA_PROJECTION_ENTRY = "camera_projection"

CHECKPOINT_NAME = "checkpoint.pth"