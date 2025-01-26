import os
import bpy

from SEDcreator import SEDcreator_utils
from SEDcreator import SEDcreator_launchRender

class RenderProperties(bpy.types.PropertyGroup):
    renderReady: bpy.props.BoolProperty(name="Toggle Option")
    exportFolder: bpy.props.StringProperty(
        name="Export Folder", description="Relative export folder", default="img")

    # Render map type management
    bool_albedo: bpy.props.BoolProperty(name="Albedo",
                                        description="Render albedo map",
                                        default=True
                                         )
    bool_depth: bpy.props.BoolProperty(name="Depth",
                                       description="Render depth map",
                                       default=True
                                        )
    bool_normal: bpy.props.BoolProperty(name="Normal",
                                        description="Render normal map",
                                        default=True
                                         )
    bool_id: bpy.props.BoolProperty(name="Id",
                                    description="Render id map",
                                    default=True
                                     )
    # Beauty is always rendered
    bool_beauty: bpy.props.BoolProperty(name="Beauty",
                                        description="Beauty render (always rendered)",
                                        default=True
                                         )
    bool_transmission: bpy.props.BoolProperty(name="Transmission",
                                              description="Transmission mask",
                                              default=True
                                               )
    bool_roughness: bpy.props.BoolProperty(name="Roughness",
                                              description="Roughness mask",
                                              default=True
                                              )
    bool_curvature: bpy.props.BoolProperty(name="Curvature",
                                           description="Curvature mask",
                                           default=True
                                              )
    # First and last frame of the rendering
    start: bpy.props.IntProperty(name="FirstFrame", default=0)
    end: bpy.props.IntProperty(name="LastFrame", default=1)

class RenderOperator(bpy.types.Operator):
    bl_idname = "object.sed_render"
    bl_label = "Start Render"
    bl_description = "Start render with the set parameters. Please, open the console to be able to follow the rendering."
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.scene.RenderProperties.renderReady == True

    def execute(self, context):

        renderProp = context.scene.RenderProperties

        # Get the img folder path
        filePath = bpy.data.filepath
        curDir = os.path.dirname(filePath)
        imgDir = os.path.join(curDir, renderProp.exportFolder)

        # Create the img folder if it does not exist
        os.makedirs(imgDir, exist_ok=True)

        # Get the dome shape
        #domeShape = context.scene.SetupProperties.domeShape

        # Renumber the cameras
        SEDcreator_utils.renumberSEDCameras(context)
        sedCameras = SEDcreator_utils.getSEDCameras()
        # Array of the cameras which render an image
        camerasObjs = sedCameras[renderProp.start:renderProp.end + 1]

        print("---------- Rendering start ----------")
        SEDcreator_launchRender.launchRender(context, camerasObjs, imgDir)
        print("---------- Rendering end ----------")

        return {'FINISHED'}

classes = [RenderProperties, RenderOperator]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.RenderProperties = bpy.props.PointerProperty(type=RenderProperties)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.RenderProperties


if __name__ == "__main__":
    register()
