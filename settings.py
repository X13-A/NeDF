DATA_PATH = "./data/robot_orbital_bas"
CAMERAS_PATH = f"{DATA_PATH}/cameras_attributes.npz"
DEPTHS_PATH = f"{DATA_PATH}/depth"

FOV = 30.0
DEPTHMAP_SIZE_RESCALE = 100/1920
SCENE_SCALE = 1/1000

# BASE DATASET STRUCTURE
BASE_CAMERA_LOCATION_ENTRY = "cameras_locations"
BASE_CAMERA_ANGLE_ENTRY = "cameras_angle"

# TARGET DATASET STRUCTURE
RAYS_ENTRY = "rays"
RAY_ORIGINS_ENTRY = "ray_origins"
RAY_DIRECTIONS_ENTRY = "ray_directions"
DEPTH_MAP_ENTRY = "depth_map"
CAMERA_POS_ENTRY = "camera_pos"
CAMERA_ANGLE_ENTRY = "camera_angle"
VALID_INDICES_ENTRY = "valid_indices"
VALID_2D_INDICES_ENTRY = "valid_2D_indices"
CAMERA_TRANSFORM_ENTRY = "camera_transform"
CAMERA_PROJECTION_ENTRY = "camera_projection"