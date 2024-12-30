import bpy
import math

# Useful functions for all the project

# Check if a string corresponds to an object of the scene
def objectNameInScene(name):
    for obj in bpy.data.objects:
        if obj.name == name:
            return True
    return False

# Create an array of SED cameras
def getSEDCameras():
    res = []
    collectionsName = ["IcoSEDCollection", "SemiIcoSEDCollection", "AdaptativeIcoSEDCollection", "SphereSEDCollection", "SemiSphereSEDCollection", "AdaptativeSphereSEDCollection"]
    for collectionName in collectionsName:
        for obj in bpy.data.collections[collectionName].objects:
            if obj.type == 'CAMERA':
                res.append(obj)
    return res

# Renumber SED cameras
def renumberSEDCameras(context):
    cameras = getSEDCameras()
    number_cam = 0
    for cam in cameras:
        # Check if the Camera_{number_cam} already existed for renumbering correctly
        new_name = f"Camera_{number_cam}"
        existing_camera = bpy.data.objects.get(new_name)
        if existing_camera:
            bpy.data.objects[new_name].name = "Camera_"
        cam.name = f"Camera_{number_cam}"
        number_cam += 1
        
# Check if an object is in a cube
def inCube(obj_location, x_min, x_max, y_min, y_max, z_min, z_max):
    return (obj_location.x >= x_min and obj_location.x <= x_max) and (obj_location.y >= y_min and obj_location.y <= y_max) and (obj_location.z >= z_min and obj_location.z <= z_max)

# Instanciate a camera
def createCamera(context, lens):
    cam = bpy.data.cameras.new("Camera")
    cam.lens_unit = lens
    cam.lens = context.scene.SetupProperties.focalLength
    return cam

def createCameraObj(context, name, cam, loc=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)):
    radiansRot = tuple([math.radians(a)for a in rot])  # Convert angles to radians
    obj = bpy.data.objects.new(name, cam)
    obj.location = loc
    obj.rotation_euler = radiansRot
    # Nothing changes but it is easier to read in the 3D Viewer like this
    obj.scale = (1, 1, 1)

    context.collection.objects.link(obj)

    # Move origin (could be improved)
    active = context.view_layer.objects.active
    context.view_layer.objects.active = obj
    context.view_layer.objects.active = active

    return obj

# Create a camera on a vertice of a shape
def createCameraOnShape(context, object, shape, cam, vertice, position):
    setup_properties = context.scene.SetupProperties
    camName = f"Camera_"
    current_cam = createCameraObj(context, camName, cam, (object.location.x + setup_properties.clusterRadius, 0, 0), (90, 0, 90))
    #to keep the cameras where they should be after parenting operation
    current_cam.parent = object
    current_cam.matrix_parent_inverse = object.matrix_world.inverted()
    current_cam.select_set(True)
    context.view_layer.objects.active = current_cam  # (could be improved)

    current_cam.location = position
    context.view_layer.update()
    if setup_properties.orientationCameras == 'I':
        lookAtIn(current_cam, shape.matrix_world.to_translation())
    elif setup_properties.orientationCameras == 'O':
        lookAtOut(current_cam, shape.matrix_world.to_translation())
    else:
        focus_object = context.scene.SetFocusProperties.focus_object
        lookAtSelect(current_cam, bpy.data.objects[focus_object])

# Delete all children of an object
def deleteChildren(object):
    children = object.children
    for child in children:
        bpy.data.objects.remove(child, do_unlink=True)

# Rotate obj_camera to look at point
def lookAtIn(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()

# Rotate obj_camera to look in opposite direction of point
def lookAtOut(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()

    direction = loc_camera - point
    rot_quat = direction.to_track_quat('-Z', 'Y')

    obj_camera.rotation_euler = rot_quat.to_euler()

# Rotate obj_camera to look at the center of object
def lookAtSelect(obj_camera, object):
    loc_camera = obj_camera.matrix_world.to_translation()
    point = object.matrix_world.to_translation()
    direction = point - loc_camera
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_camera.rotation_euler = rot_quat.to_euler()

def replaceMaterialByOriginal(object):
    if object.material_slots:
        mat = object.material_slots[0].material
        tree = mat.node_tree
        nodes = tree.nodes
        
        # Get Principled node
        principled = nodes['Principled BSDF']
        
        # Get Material Output node
        material_output = nodes['Material Output']
        
        # Change link
        tree.links.new(principled.outputs['BSDF'], material_output.inputs['Surface'])

def removeAttributeFromAllMaterials():
    materials = bpy.data.materials
    for mat in materials:
        if mat.node_tree:
            # Get Material Output node
            material_output = mat.node_tree.nodes['Material Output']
            mo_sockets = material_output.inputs
            
            # Get the node that is connected to the Material Output
            link = mo_sockets['Surface'].links[0]
            link_nodes = link.from_node
            
            # If this node is Attribute, remove it
            if link_nodes.outputs[0].name == "Color":
                mat.node_tree.nodes.remove(mat.node_tree.nodes['Attribute'])

def replaceMaterialByCurvature(object):
    if object.material_slots:
        mat = object.material_slots[0].material
        tree = mat.node_tree
        nodes = tree.nodes

        # Get Attribute node
        attribute = nodes['Attribute']
        
        # Get Material Output node
        material_output = nodes['Material Output']
        
        # Change link
        tree.links.new(attribute.outputs['Color'], material_output.inputs['Surface'])


def addAttributeToAllMaterials():
    materials = bpy.data.materials
    for mat in materials:
        if mat.node_tree:
            attribute = mat.node_tree.nodes.new("ShaderNodeAttribute")
            attribute.attribute_type = 'GEOMETRY'
            attribute.attribute_name = "Curvature"

def replaceMaterialByRoughness(object):
    if object.material_slots:
        mat = object.material_slots[0].material
        tree = mat.node_tree
        nodes = tree.nodes
        
        # Get Principled node
        principled = nodes['Principled BSDF']
        p_sockets = principled.inputs
        
        # Get Roughness image node, if it exists
        if p_sockets['Roughness'].links:
            p_link_roughness = p_sockets['Roughness'].links[0]
            roughness = p_link_roughness.from_node
            
            # Get Material Output node
            material_output = nodes['Material Output']
            
            # Change link
            tree.links.new(roughness.outputs['Color'], material_output.inputs['Surface'])

# Initialize curvature node group
def curvatureNodeGroup():
    curvature = bpy.data.node_groups.new(type = 'GeometryNodeTree', name = "VertexGroupWriterFromCurvature")
    curvature.is_tool = True
    curvature.is_modifier = True

    # Initialize curvature nodes
    
    # Segmentation inputs
    # Input Geometry
    curvature.interface.new_socket(name = 'Geometry', in_out ="INPUT", socket_type = "NodeSocketGeometry")
    curvature.interface.items_tree[0].attribute_domain = 'POINT'

    # Node Group Input
    group_input = curvature.nodes.new("NodeGroupInput")
    
    # Node Math - Divide
    normalize = curvature.nodes.new("ShaderNodeMath")
    normalize.operation = 'DIVIDE'
    # Value_001
    normalize.inputs[1].default_value = 3.16

    # Node Math - Multiply
    eraser = curvature.nodes.new("ShaderNodeMath")
    eraser.operation = 'MULTIPLY'
    # Value_002
    eraser.inputs[1].default_value = 0

    # Node Math - Add
    half_of_one = curvature.nodes.new("ShaderNodeMath")
    half_of_one.operation = 'ADD'
    # Value_002
    half_of_one.inputs[1].default_value = 0.5
    
    # Node Named Attribute
    named_attribute = curvature.nodes.new("GeometryNodeInputNamedAttribute")
    named_attribute.inputs[0].default_value = "Curvature"
    
    # Node Math - Add
    math_add = curvature.nodes.new("ShaderNodeMath")
    math_add.operation = 'ADD'
    math_add.use_clamp = True
    
    # Node store named attribute
    store_attribute = curvature.nodes.new("GeometryNodeStoreNamedAttribute")
    store_attribute.data_type = 'FLOAT'
    store_attribute.domain = 'POINT'
    store_attribute.inputs["Name"].default_value = "Curvature"
    
    # Node Edge Angle
    edge_angle = curvature.nodes.new("GeometryNodeInputMeshEdgeAngle")
    
    # Segmentation outputs
    # Output Geometry
    curvature.interface.new_socket(name = 'Geometry', in_out ="OUTPUT", socket_type = "NodeSocketGeometry")
    curvature.interface.items_tree[0].attribute_domain = 'POINT'

    # Node Group Output
    group_output = curvature.nodes.new("NodeGroupOutput")

    # Set locations
    group_input.location = (-214.06039428710938, 73.0862045288086)
    store_attribute.location = (29.041637420654297, 58.562652587890625)
    group_output.location = (270.24505615234375, -11.900897979736328)
    named_attribute.location = (-518.30517578125, -93.80853271484375)
    math_add.location = (-282.4539794921875, -164.74874877929688)
    edge_angle.location = (-844.835693359375, -334.1424255371094)
    normalize.location = (-655.8768310546875, -303.60455322265625)
    half_of_one.location = (-470.9601135253906, -294.2790222167969)

    # Set dimensions
    group_input.width, group_input.height = 140.0, 100.0
    normalize.width, normalize.height = 140.0, 100.0
    half_of_one.width, half_of_one.height = 140.0, 100.0
    edge_angle.width, edge_angle.height = 140.0, 100.0
    group_output.width, group_output.height = 140.0, 100.0

    # Initialize curvature links
    # Group_input.Geometry -> store_attribute.Geometry
    curvature.links.new(group_input.outputs['Geometry'], store_attribute.inputs['Geometry'])
    # Store_attribute.Geometry -> group_output.Geometry
    curvature.links.new(store_attribute.outputs['Geometry'], group_output.inputs['Geometry'])
    # Edge_angle.Signed Angle -> normalize.Value
    curvature.links.new(edge_angle.outputs['Signed Angle'], normalize.inputs['Value'])
    # Narmalize.Value -> half_of_one.Value
    curvature.links.new(normalize.outputs['Value'], half_of_one.inputs['Value'])
    # Half_of_one.Value -> math_add.Value
    curvature.links.new(half_of_one.outputs['Value'], math_add.inputs[1])
    # Named_attribute.Curvature -> math_add.Value
    curvature.links.new(named_attribute.outputs['Attribute'], eraser.inputs[0])
    curvature.links.new(eraser.outputs['Value'], math_add.inputs[0])
    # Math_add.Value -> store_attribute
    curvature.links.new(math_add.outputs['Value'], store_attribute.inputs['Value'])
    return curvature


def addVertexGroup(object, name):
    if object.type == 'MESH':
        vertex_groups = object.vertex_groups
        vertex_groups.new(name = name)
    else:
        return
    
    
def applyModifier(object, modifier):
    seg_modifier = object.modifiers.new(name = "GeometryNode", type = 'NODES')
    if seg_modifier is not None:
        seg_modifier.node_group = modifier
    else:
        return
