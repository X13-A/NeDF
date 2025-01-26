import bpy
import os
import numpy as np
from SEDcreator import SEDcreator_utils
import mathutils
import bpy

class SaveCamerasAttributesOperator(bpy.types.Operator):
    bl_idname = "object.sed_save_cameras_attributes"
    bl_label = "Save Cameras Attributes"
    bl_description = "Go through all cameras created with this add-on and save their location, angle and focal length to a .npz file"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.RenderProperties.renderReady == True # on le met quand meme car s'il n'y a pas de cameras ca sert a rien

    def execute(self, context):
        # Get the img folder path
        renderProp = context.scene.RenderProperties
        filePath = bpy.data.filepath
        curDir = os.path.dirname(filePath)
        imgDir = os.path.join(curDir, renderProp.exportFolder)

        # Create the img folder if it does not exist
        os.makedirs(imgDir, exist_ok=True)

        # Renumber the cameras
        SEDcreator_utils.renumberSEDCameras(context)
        sedCameras = SEDcreator_utils.getSEDCameras()
        camerasObjs = sedCameras[renderProp.start:renderProp.end + 1]

        print("---------- Save cameras attributes start ----------")
        cameras_locations, cameras_angles, ray_maps = self.launchCamerasAttributes(camerasObjs)
        self.saveCamerasAttributes(imgDir, cameras_locations, cameras_angles, ray_maps)

        print("---------- Save cameras attributes end ----------")

        return {'FINISHED'}

    def saveCamerasAttributes(self, imgDir, cams_locations, cams_angle, ray_maps):
        cameras_attributes_file = os.path.join(imgDir, "cameras_attributes.npz")
        np.savez(cameras_attributes_file, 
                cameras_locations=cams_locations, 
                cameras_angle=cams_angle, 
                ray_maps=ray_maps)

    def launchCamerasAttributes(self, camerasObjs):
        cameras_locations = []
        cameras_angle = []
        ray_maps = []

        for cam in camerasObjs:
            # Store camera location and angle
            cameras_locations.append(cam.location)
            cameras_angle.append(cam.rotation_euler)

            # Compute ray map for the camera
            ray_map = self.computeRayMap(cam)
            ray_maps.append(ray_map)

        cameras_locations = np.array(cameras_locations)
        cameras_angle = np.array(cameras_angle)
        ray_maps = np.array(ray_maps)

        return cameras_locations, cameras_angle, ray_maps

    def computeRayMap(self, cam):
        """Compute the ray map for the given camera."""
        cam_data = cam.data

        # Camera resolution and properties
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
    #bpy.types.Scene.SaveCamerasAttributesProperties = bpy.props.PointerProperty(type=SaveCamerasAttributesProperties)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    #del bpy.types.Scene.SaveCamerasAttributesProperties


if __name__ == "__main__":
    register()
