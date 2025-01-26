import os
import bpy
import glob
from SEDcreator import SEDcreator_utils

# Functions to prepare the rendering
def prepareRenderBeauty(context, imgDir, imgName):
        path_beauty = os.path.join(imgDir, "beauty", imgName)
        context.scene.render.filepath = path_beauty

def prepareRenderDepth(bool_depth, nodes, links, format, color_depth, render_layers, imgDir, imgName):
    if bool_depth:
        # Create depth output nodes
        depth_file_output = nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        depth_file_output.base_path = "" 
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = format
        depth_file_output.format.color_depth = color_depth
        if format == "OPEN_EXR":
            links.new(render_layers.outputs["Depth"],
                      depth_file_output.inputs[0])
        else:
            depth_file_output.format.color_mode = "BW"

            # Remap as other types can not represent the full range of depth.
            map = nodes.new(type="CompositorNodeMapValue")
            # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
            map.offset = [-0.7]
            map.size = [1.4]
            map.use_min = True
            map.min = [0]
            links.new(render_layers.outputs["Depth"], map.inputs[0])
            links.new(map.outputs[0], depth_file_output.inputs[0])

        path_depth = os.path.join(imgDir, "depth" , imgName)
        depth_file_output.file_slots[0].path = path_depth +  "_depth"

def prepareRenderNormal(bool_normal, nodes, links, format, render_layers, imgDir, imgName):
    if bool_normal:
        # Create normal output nodes
        scale_node = nodes.new(type="CompositorNodeMixRGB")
        scale_node.blend_type = "MULTIPLY"
        # scale_node.use_alpha = True
        scale_node.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs["Normal"], scale_node.inputs[1])

        bias_node = nodes.new(type="CompositorNodeMixRGB")
        bias_node.blend_type = "ADD"
        # bias_node.use_alpha = True
        bias_node.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        links.new(scale_node.outputs[0], bias_node.inputs[1])

        normal_file_output = nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = "Normal Output"
        normal_file_output.base_path = ''
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = format
        links.new(bias_node.outputs[0], normal_file_output.inputs[0])

        path_normal = os.path.join(imgDir, "normal", imgName)
        normal_file_output.file_slots[0].path = path_normal + "_normal"

def prepareRenderAlbedo(bool_albedo, nodes, links, format, color_depth, render_layers, imgDir, imgName):
    if bool_albedo:
        # Create albedo output nodes
        alpha_albedo = nodes.new(type="CompositorNodeSetAlpha")
        links.new(render_layers.outputs['DiffCol'],
                  alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])

        albedo_file_output = nodes.new(type="CompositorNodeOutputFile")
        albedo_file_output.label = 'Albedo Output'
        albedo_file_output.base_path = ''
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = format
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = color_depth
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])

        path_albedo = os.path.join(imgDir, "albedo", imgName)
        albedo_file_output.file_slots[0].path = path_albedo + "_albedo"

def prepareRenderId(bool_id, nodes, links, format, color_depth, render_layers, imgDir, imgName):
    if bool_id:
        # Create id map output nodes
        id_file_output = nodes.new(type="CompositorNodeOutputFile")
        id_file_output.label = 'ID Output'
        id_file_output.base_path = ''
        id_file_output.file_slots[0].use_node_format = True
        id_file_output.format.file_format = format
        id_file_output.format.color_depth = color_depth

        if format == 'OPEN_EXR':
            links.new(
                render_layers.outputs['IndexOB'], id_file_output.inputs[0])
        else:
            id_file_output.format.color_mode = 'BW'

            divide_node = nodes.new(type='CompositorNodeMath')
            divide_node.operation = 'DIVIDE'
            divide_node.use_clamp = False
            divide_node.inputs[1].default_value = 2 ** int(color_depth)

            links.new(render_layers.outputs['IndexOB'], divide_node.inputs[0])
            links.new(divide_node.outputs[0], id_file_output.inputs[0])

        path_id = os.path.join(imgDir, "id", imgName)
        id_file_output.file_slots[0].path = path_id + "_id"

def prepareRenderTransmission(bool_transmission, nodes, links, format, color_depth, render_layers, imgDir, imgName):
    if bool_transmission:
        transmission_file_output = nodes.new("CompositorNodeOutputFile")
        transmission_file_output.label = "Transmission Output"
        transmission_file_output.base_path = ""
        transmission_file_output.file_slots[0].use_node_format = True
        transmission_file_output.format.file_format = format
        transmission_file_output.format.color_depth = color_depth
        links.new(render_layers.outputs["TransCol"], transmission_file_output.inputs[0])

        path_transmission = os.path.join(imgDir, "transmission", imgName)
        transmission_file_output.file_slots[0].path = path_transmission + "_transmission"

def prepareRenderCurvature(bool_curvature, context, imgDir, imgName):
    if bool_curvature:
        # Create the node tree for our new modifier
        curvature = SEDcreator_utils.curvatureNodeGroup()
        
        # Get a list of all objects (selected)
        bpy.ops.object.select_all(action='SELECT')
        selected = context.selected_objects
        
        # Create and apply our new modifier to all these objects
        for obj in selected:
            if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                SEDcreator_utils.addVertexGroup(obj, "Curvature")
                SEDcreator_utils.applyModifier(obj, curvature)
        
        # Add Attibute node to all materials
        SEDcreator_utils.addAttributeToAllMaterials()
        
        # Replace all material for these objects by their curvature map
        for obj in selected:
            if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                SEDcreator_utils.replaceMaterialByCurvature(obj)

        # Deselect all objects    
        bpy.ops.object.select_all(action='DESELECT')

        path_curvature = os.path.join(imgDir, "curvature", imgName)
        context.scene.render.filepath = path_curvature + "_curvature"

def prepareRenderRoughness(bool_roughness, context, imgDir, imgName):
    if bool_roughness:
        # Get a list of all objects (selected)
        bpy.ops.object.select_all(action='SELECT')
        selected = context.selected_objects

        # Replace all material for these objects by their roughness image
        for obj in selected:
            # Let the lights just in case
            if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                SEDcreator_utils.replaceMaterialByRoughness(obj)

        # Deselect all objects    
        bpy.ops.object.select_all(action='DESELECT')

        path_roughness = os.path.join(imgDir, "roughness", imgName)
        context.scene.render.filepath = path_roughness + "_roughness"

def enableUsePasses(context):
    # Enable by default all useful use_pass
    context.scene.use_nodes = True
    context.scene.view_layers["ViewLayer"].use_pass_z = True
    context.scene.view_layers["ViewLayer"].use_pass_normal = True
    context.scene.view_layers["ViewLayer"].use_pass_diffuse_color = True
    context.scene.view_layers["ViewLayer"].use_pass_object_index = True
    context.scene.view_layers["ViewLayer"].use_pass_transmission_color = True

def replaceObjectsByOriginals(renderType, context):
    renderProp = context.scene.RenderProperties
    if (renderType == "Roughness" and renderProp.bool_roughness) or (renderType == "Curvature" and renderProp.bool_curvature):
        # Get a list of all objects (selected)
        bpy.ops.object.select_all(action='SELECT')
        selected = context.selected_objects

        if renderType == "Curvature":
            # Remove Attribute node from all materials
            SEDcreator_utils.removeAttributeFromAllMaterials()

        # Replace all material for these objects by their original texture images
        for obj in selected:
            # Let the lights just in case
            if obj.type != 'CAMERA' and obj.type != 'LIGHT':
                SEDcreator_utils.replaceMaterialByOriginal(obj)
        # Deselect all objects    
        bpy.ops.object.select_all(action='DESELECT')

# Rename correctly all output files (without the xxxx at the end)
def renameFiles(imgDir, imgName):
    types = ["depth", "normal", "albedo", "id", "transmission", "curvature", "roughness"]
    for s in types:
        dir = os.path.join(imgDir, s)
        path_depth = os.path.join(dir, imgName)
        list = glob.glob(path_depth + "*")
        for l in list:
            os.rename(l, path_depth + "_" + s)

