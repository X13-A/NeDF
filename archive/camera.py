import glm
from transform import Transform

class Camera:
    def __init__(self, position=glm.vec3(0.0, 0.0, 0.0), yaw=0.0, pitch=0.0, roll=0.0, fov=45.0, aspect_ratio=16.0/9.0, near=0.1, far=100.0):
        self.transform = Transform(position=position, rotation=glm.vec3(pitch, yaw, roll))
        self.projection_matrix = glm.perspective(glm.radians(fov), aspect_ratio, near, far)
        self.update_view_matrix()
        
    def update_view_matrix(self):
        # Calculate the direction vector from yaw and pitch
        yaw = self.transform.rotation.y
        pitch = self.transform.rotation.x
        direction = glm.vec3(
            glm.cos(glm.radians(yaw)) * glm.cos(glm.radians(pitch)),
            glm.sin(glm.radians(pitch)),
            glm.sin(glm.radians(yaw)) * glm.cos(glm.radians(pitch))
        )
        direction = glm.normalize(direction)

        self.view_matrix = glm.lookAt(self.transform.position, self.transform.position + direction, glm.vec3(0.0, 1.0, 0.0))

    def translate(self, delta):
        # Translate the camera and update the view matrix
        self.transform.translate(delta)
        self.update_view_matrix()

    def rotate(self, delta_yaw, delta_pitch, delta_roll):
        # Rotate the camera and update the view matrix
        self.transform.rotate(delta_yaw, delta_pitch, delta_roll)
        self.update_view_matrix()