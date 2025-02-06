import bpy
import os
import numpy as np
from SEDcreator import SEDcreator_utils
import mathutils

class SaveCamerasAttributesOperator(bpy.types.Operator):
    bl_idname = "object.sed_save_cameras_attributes"
    bl_label = "Save Cameras Attributes"
    bl_description = "Save camera location, angles, transformation matrix, projection matrix, and ray map to a .npz file"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.RenderProperties.renderReady == True  # Ensure render is ready

    def execute(self, context):
        # Get export folder
        renderProp = context.scene.RenderProperties
        filePath = bpy.data.filepath
        curDir = os.path.dirname(filePath)
        imgDir = os.path.join(curDir, renderProp.exportFolder)

        # Create directory if it does not exist
        os.makedirs(imgDir, exist_ok=True)

        # Renumber cameras and get list
        SEDcreator_utils.renumberSEDCameras(context)
        sedCameras = SEDcreator_utils.getSEDCameras()
        camerasObjs = sedCameras[renderProp.start:renderProp.end + 1]

        print("---------- Save cameras attributes start ----------")
        cameras_locations, cameras_angles, cameras_matrices, projection_matrices, ray_maps = self.launchCamerasAttributes(camerasObjs)
        self.saveCamerasAttributes(imgDir, cameras_locations, cameras_angles, cameras_matrices, projection_matrices, ray_maps)
        print("---------- Save cameras attributes end ----------")

        return {'FINISHED'}

    def saveCamerasAttributes(self, imgDir, cams_locations, cams_angle, cams_matrices, projection_matrices, ray_maps):
        """
        Save camera attributes to .npz file.
        """
        cameras_attributes_file = os.path.join(imgDir, "cameras_attributes.npz")
        np.savez(cameras_attributes_file, 
                 cameras_locations=cams_locations, 
                 cameras_angle=cams_angle, 
                 cameras_matrices=cams_matrices, 
                 projection_matrices=projection_matrices,  # New: Projection matrices
                 ray_maps=ray_maps)

    def launchCamerasAttributes(self, camerasObjs):
        """
        Get camera attributes: location, rotation, transformation matrix, projection matrix, and ray map.
        """
        cameras_locations = []
        cameras_angles = []
        cameras_matrices = []
        projection_matrices = []
        ray_maps = []

        for cam in camerasObjs:
            # Store position, rotation, and transformation matrix
            cameras_locations.append(cam.location)
            cameras_angles.append(cam.rotation_euler)
            cameras_matrices.append(np.array(cam.matrix_world))  # Convert to numpy array

            # Compute projection matrix
            projection_matrix = self.computeProjectionMatrix(cam)
            projection_matrices.append(projection_matrix)

            # Compute ray map
            ray_map = self.computeRayMap(cam)
            ray_maps.append(ray_map)

        cameras_locations = np.array(cameras_locations)
        cameras_angles = np.array(cameras_angles)
        cameras_matrices = np.array(cameras_matrices)
        projection_matrices = np.array(projection_matrices)  # Convert list to numpy array
        ray_maps = np.array(ray_maps)

        return cameras_locations, cameras_angles, cameras_matrices, projection_matrices, ray_maps


    def computeProjectionMatrix(self, cam):
        """
        Get the camera's projection matrix directly from Blender.
        """
        scene = bpy.context.scene
        depsgraph = bpy.context.evaluated_depsgraph_get()

        width = scene.render.resolution_x
        height = scene.render.resolution_y

        # Fix: Use keyword arguments (x=width, y=height)
        projection_matrix = cam.calc_matrix_camera(depsgraph, x=width, y=height)

        return np.array(projection_matrix)


    def computeRayMap(self, cam):
        """
        Compute the ray map for the given camera.
        """
        cam_data = cam.data

        # Get camera properties
        scene = bpy.context.scene
        width = scene.render.resolution_x
        height = scene.render.resolution_y
        aspect_ratio = width / height

        # Camera intrinsics
        sensor_width = cam_data.sensor_width
        sensor_height = cam_data.sensor_height / aspect_ratio if cam_data.sensor_fit == 'HORIZONTAL' else cam_data.sensor_height
        focal_length = cam_data.lens

        # Create ray directions
        directions = np.zeros((height, width, 3), dtype=np.float32)
        camera_matrix_world = cam.matrix_world

        for y in range(height):
            for x in range(width):
                # Pixel normalized coordinates
                nx = (x + 0.5) / width - 0.5
                ny = 0.5 - (y + 0.5) / height

                # Convert to camera coordinates
                cam_x = nx * sensor_width
                cam_y = ny * sensor_height
                cam_z = -focal_length

                # Transform to world coordinates
                direction = mathutils.Vector((cam_x, cam_y, cam_z))
                direction_world = camera_matrix_world.to_3x3() @ direction
                direction_world.normalize()

                directions[y, x] = [direction_world.x, direction_world.y, direction_world.z]

        return directions

classes = [SaveCamerasAttributesOperator]

def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
