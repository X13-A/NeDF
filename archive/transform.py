import glm

class Transform:
    def __init__(self, position=None, rotation=None, scale=None):
        self.position = position if position is not None else glm.vec3(0.0, 0.0, 0.0)
        self.rotation = rotation if rotation is not None else glm.vec3(0.0, 0.0, 0.0)  # Represented as (yaw, pitch, roll)
        self.scale = scale if scale is not None else glm.vec3(1.0, 1.0, 1.0)

        # Store the transformation matrix, which needs to be updated when position, rotation, or scale changes
        self.update_transform_matrix()

    def update_transform_matrix(self):
        # Translation matrix
        translation_matrix = glm.translate(glm.mat4(1.0), self.position)

        # Rotation matrices for yaw (Y-axis), pitch (X-axis), and roll (Z-axis)
        yaw_matrix = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.y), glm.vec3(0.0, 1.0, 0.0))
        pitch_matrix = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.x), glm.vec3(1.0, 0.0, 0.0))
        roll_matrix = glm.rotate(glm.mat4(1.0), glm.radians(self.rotation.z), glm.vec3(0.0, 0.0, 1.0))

        # Combined rotation matrix (in Yaw-Pitch-Roll order)
        rotation_matrix = yaw_matrix * pitch_matrix * roll_matrix

        # Scale matrix
        scale_matrix = glm.scale(glm.mat4(1.0), self.scale)

        # Combine all transformations: Translation * Rotation * Scale
        self.transform_matrix = translation_matrix * rotation_matrix * scale_matrix

    def translate(self, delta):
        self.position += delta
        self.update_transform_matrix()

    def rotate(self, delta_yaw, delta_pitch, delta_roll):
        self.rotation.y += delta_yaw   # Yaw (Y-axis)
        self.rotation.x += delta_pitch # Pitch (X-axis)
        self.rotation.z += delta_roll  # Roll (Z-axis)
        self.update_transform_matrix()

    def scale_transform(self, delta_scale):
        self.scale *= delta_scale
        self.update_transform_matrix()