import os
import bpy
from SEDcreator import SEDcreator_prepareRender

# Render function for all except roughness and curvature
def render(context, imgDir, imgName):
    # Set up rendering
    scene = context.scene
    #render = scene.render
    renderProp = scene.RenderProperties

    SEDcreator_prepareRender.enableUsePasses(context)

    nodes = scene.node_tree.nodes
    links = scene.node_tree.links

    # Clear default nodes
    for n in nodes:
        nodes.remove(n)

    # Create input render layer node
    render_layers = nodes.new('CompositorNodeRLayers')

    format = "OPEN_EXR"
    color_depth = "16"

    os.chdir("//")

    SEDcreator_prepareRender.prepareRenderBeauty(context, imgDir, imgName)
    SEDcreator_prepareRender.prepareRenderDepth(renderProp.bool_depth, nodes, links, format, color_depth, render_layers, imgDir, imgName)
    SEDcreator_prepareRender.prepareRenderNormal(renderProp.bool_normal, nodes, links, format, render_layers, imgDir, imgName)
    SEDcreator_prepareRender.prepareRenderAlbedo(renderProp.bool_albedo, nodes, links, format, color_depth, render_layers, imgDir, imgName)
    SEDcreator_prepareRender.prepareRenderId(renderProp.bool_id, nodes, links, format, color_depth, render_layers, imgDir, imgName)
    SEDcreator_prepareRender.prepareRenderTransmission(renderProp.bool_transmission, nodes, links, format, color_depth, render_layers, imgDir, imgName)

    bpy.ops.render.render(write_still=True)  # render still

    return
