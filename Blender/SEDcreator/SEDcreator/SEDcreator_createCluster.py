import bpy

from SEDcreator import SEDcreator_utils

# Creation of the cluster of cameras and link them to the associated empty
def create(context, object):
    setup_properties = context.scene.SetupProperties
    centerCluster = object.location
    domeShape = setup_properties.domeShape
    x_min = setup_properties.x_min
    x_max = setup_properties.x_max
    y_min = setup_properties.y_min
    y_max = setup_properties.y_max
    z_min = setup_properties.z_min
    z_max = setup_properties.z_max

    SEDcreator_utils.deleteChildren(object)

    # Create a shape in function of the selected option
    if domeShape == 'I' or domeShape == 'SI' or domeShape == 'AI':
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=setup_properties.nbSubdiv, radius=setup_properties.clusterRadius, calc_uvs=True, enter_editmode=False, align='WORLD', location=centerCluster, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(segments=setup_properties.nbSegment, ring_count=setup_properties.nbRing, radius=setup_properties.clusterRadius, calc_uvs=True, enter_editmode=False, align='WORLD', location=centerCluster, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0))

    shape = context.selected_objects[0]
    shape.name = "shape_cluster"
    #nbVertices = len(shape.data.vertices) # surement à enlever
    cam = SEDcreator_utils.createCamera(context, 'MILLIMETERS')

    # Place a camera on each vertices of the shape
    for (i, _) in enumerate(shape.data.vertices):
        v = shape.data.vertices[i]
        co_final = shape.matrix_world @ v.co

        if  (domeShape == 'SI'or domeShape == 'SS') and (object.location.z - co_final.z <= 0):
            SEDcreator_utils.createCameraOnShape(context, object, shape, cam, v, co_final)
        if (domeShape == 'AI' or domeShape == 'AS') and SEDcreator_utils.inCube(co_final, x_min, x_max, y_min, y_max, z_min, z_max):
            SEDcreator_utils.createCameraOnShape(context, object, shape, cam, v, co_final)
        if domeShape == 'I' or domeShape == 'U':
            SEDcreator_utils.createCameraOnShape(context, object, shape, cam, v, co_final)

    # Delete the shape
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects['shape_cluster'].select_set(True)
    bpy.ops.object.delete()

    print("Done")
