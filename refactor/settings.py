DATA_PATH = "./data/alex"
CAMERAS_PATH = f"{DATA_PATH}/cameras_attributes.npz"
DEPTHS_PATH = f"{DATA_PATH}/depth"
RAYS_PATH = f"{DATA_PATH}/rays"

FOV = 60
DEPTHMAP_SIZE_RESCALE = 100/1920
TARGET_IMAGE_SIZE = [100, 100]

# BASE DATASET STRUCTURE
BASE_CAMERA_LOCATION_ENTRY = "cameras_locations"
BASE_CAMERA_ANGLE_ENTRY = "cameras_angle"
BASE_CAMERA_FOV_ENTRY = "cameras_fovs"
BASE_CAMERA_FOCAL_LENGTH_ENTRY = "cameras_focal_lengths"
BASE_CAMERA_RAYS_ENTRY = "ray_maps"

# TARGET DATASET STRUCTURE
RAYS_ENTRY = "rays"
RAY_ORIGINS_ENTRY = "ray_origins"
RAY_DIRECTIONS_ENTRY = "ray_directions"
RAY_DIRECTIONS_BLENDER_ENTRY = "ray_directions_blender"
DEPTH_MAP_ENTRY = "depth_map"
FILTERED_DEPTH_MAP_ENTRY = "filtered_depth_map"
CAMERA_POS_ENTRY = "camera_pos"
CAMERA_POS_BLENDER_ENTRY = "camera_pos_blender"
CAMERA_FORWARD_ENTRY = "camera_forward"
CAMERA_ANGLE_BLENDER_ENTRY = "camera_angle_blender"
VALID_INDICES_ENTRY = "valid_indices"
VALID_2D_INDICES_ENTRY = "valid_2D_indices"
CAMERA_TRANSFORM_ENTRY = "camera_transform"
CAMERA_PROJECTION_ENTRY = "camera_projection"

CHECKPOINT_NAME = "checkpoint.pth"