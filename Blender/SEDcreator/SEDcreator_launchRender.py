import bpy
import numpy as np
from SEDcreator import SEDcreator_prepareRender
from SEDcreator import SEDcreator_render
from SEDcreator import SEDcreator_renderRoughness
from SEDcreator import SEDcreator_renderCurvature

def launchRender(context, camerasObjs, imgDir):
    renderProp = context.scene.RenderProperties
    frame = renderProp.start
    for cam in camerasObjs:
        # At one frame corresponds one image of a camera
        context.scene.frame_set(frame)
        cam_data = bpy.data.cameras.new("cam_render")
        cam_obj = bpy.data.objects.new("cam_render", cam_data)
        cam_obj = cam
        cam_obj.name = f"cam_render_{frame}"
        context.scene.camera = cam_obj
        if renderProp.bool_roughness:
            SEDcreator_renderRoughness.renderRoughness(context, imgDir, f"{cam_obj.name}")
            SEDcreator_prepareRender.replaceObjectsByOriginals("Roughness", context)
        if renderProp.bool_curvature:
            SEDcreator_renderCurvature.renderCurvature(context, imgDir, f"{cam_obj.name}")
            SEDcreator_prepareRender.replaceObjectsByOriginals("Curvature", context)
        SEDcreator_render.render(context, imgDir, f"{cam_obj.name}")

        # Renaming correctly output files
        SEDcreator_prepareRender.renameFiles(imgDir, f"{cam_obj.name}")

        cam_obj.name = f"Camera_{frame}"
        frame+=1
