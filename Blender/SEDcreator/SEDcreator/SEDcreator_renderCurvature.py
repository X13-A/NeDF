import os
import bpy
from SEDcreator import SEDcreator_prepareRender

def renderCurvature(context, imgDir, imgName):

    # Set up rendering
    scene = context.scene
    render = scene.render
    renderProp = scene.RenderProperties

    SEDcreator_prepareRender.enableUsePasses(context)

    nodes = scene.node_tree.nodes
    #links = scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    os.chdir("//")

    SEDcreator_prepareRender.prepareRenderCurvature(renderProp.bool_curvature, context, imgDir, imgName)

    bpy.ops.render.render(write_still=True)  # render still
    return
