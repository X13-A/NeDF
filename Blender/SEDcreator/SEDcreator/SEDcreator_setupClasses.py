import bpy

from SEDcreator import SEDcreator_createCluster
from SEDcreator import SEDcreator_utils


class SetupOperator(bpy.types.Operator):
    bl_idname = "object.sed_setup"
    bl_label = "Set Project Setup"
    bl_description = "Setup the scene with all the selected empty objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected = context.selected_objects

        # Set Cycles as the default renderer
        rdr = context.scene.render
        cle = context.scene.cycles

        rdr.engine = 'CYCLES'
        cle.device = 'CPU'

        # Setup the SED collections
        self.createSEDCollections(context)
        focus_object = context.scene.SetFocusProperties.focus_object
        if context.scene.SetupProperties.orientationCameras == 'F' and not SEDcreator_utils.objectNameInScene(focus_object):
            self.report({'ERROR'}, "You have no selected objects, please select one object")
        else:
            for obj in selected:
                if obj.type == 'EMPTY':
                    SEDcreator_createCluster.create(context, obj)
                    self.linkEmptyToCollection(obj, context)

        context.scene.RenderProperties.renderReady = True  # Set rendering Ready

        # Renumber the SED cameras
        SEDcreator_utils.renumberSEDCameras(context)

        return {'FINISHED'}

    def linkEmptyToCollection(self, object, context):
        if context.scene.SetupProperties.domeShape == 'I':
            self.linkObjectHierarchyToCollection(object, context, "IcoSEDCollection")
        elif context.scene.SetupProperties.domeShape == 'SI':
            self.linkObjectHierarchyToCollection(object, context, "SemiIcoSEDCollection")
        elif context.scene.SetupProperties.domeShape == 'AI':
            self.linkObjectHierarchyToCollection(object, context, "AdaptativeIcoSEDCollection")
        elif context.scene.SetupProperties.domeShape == 'U':
            self.linkObjectHierarchyToCollection(object, context, "SphereSEDCollection")
        elif context.scene.SetupProperties.domeShape == 'SS':
            self.linkObjectHierarchyToCollection(object, context, "SemiSphereSEDCollection")
        else:
            self.linkObjectHierarchyToCollection(object, context, "AdaptativeSphereSEDCollection")

    def linkObjectHierarchyToCollection(self, object, context, collectionName):
        children = object.children_recursive
        for child in children:
            if child.users_collection[0].name == 'Scene Collection':
                context.collection.objects.unlink(child)
            else:
                bpy.data.collections[child.users_collection[0].name].objects.unlink(child)
            bpy.data.collections[collectionName].objects.link(child)
        if object.users_collection[0].name == 'Scene Collection':
            context.collection.objects.unlink(object)
        else:
            bpy.data.collections[object.users_collection[0].name].objects.unlink(object)
        bpy.data.collections[collectionName].objects.link(object)

    def collectionExists(self, context, collectionName):
        collections = context.scene.collection.children
        for collection in collections:
            if collection.name == collectionName:
                return True
        return False

    def createSEDCollections(self, context):
        collectionsName = ["IcoSEDCollection", "SemiIcoSEDCollection", "AdaptativeIcoSEDCollection", "SphereSEDCollection", "SemiSphereSEDCollection", "AdaptativeSphereSEDCollection"]
        for collectionName in collectionsName:
            if not self.collectionExists(context, collectionName):
                collection = bpy.data.collections.new(collectionName)
                context.scene.collection.children.link(collection)

class SetupProperties(bpy.types.PropertyGroup):

    renderReady: bpy.props.BoolProperty(name="Toggle Option")
    domeShape: bpy.props.EnumProperty(name='Shape type', description='Choose the shape of the camera dome',
                                      items={
                                          ('I', 'Icosahedron',
                                            'Place the cameras along the vertices of an Icosahedron'),
                                          ('SI', 'Semi Icosahedron',
                                            'Place the cameras along the vertices of a Icosahedron dome'),
                                          ('U', 'UV Sphere',
                                           'Place the cameras along the vertices of an UV Sphere'),
                                          ('SS', 'Semi Sphere',
                                          'Place the cameras along the vertices of a dome'),
                                          ('AI', 'Adaptative Icosahedron',
                                           'Place the cameras along the vertices of a Icosahedron dome that is limited by the bounding box'),
                                          ('AS', 'Adaptative UV Sphere',
                                           'Place the cameras along the vertices of a UV Sphere that is limited by the bounding box')
                                      }, default='I')

    orientationCameras: bpy.props.EnumProperty(name="Cameras Orientation",
                                       description="Inward or outward orientation of cameras",
                                       items={
                                               ('I', 'Inward', 'Orient cameras inward'),
                                               ('O', 'Outward', 'Orient cameras outward'),
                                               ('F', 'Focus', 'Orient cameras to focus on the selectioned object'),
                                             }, default='I')

    clusterRadius: bpy.props.FloatProperty(name="Cluster radius",
                                          description="Radius of the cluster of cameras", default=1,
                                          min=0, max=10)  # In meters

    # - Icosahedron and Semi Icosahedron
    nbSubdiv: bpy.props.IntProperty(name="Number of Subdivisions", description="Number of dome shape's subdivisions",
                                    default=1, min=1, max=3, step=1)

    # - UV Sphere and Semi Sphere
    nbSegment: bpy.props.IntProperty(name="Number of segments", description="Number of sphere's rings", default=16,
                                     min=3, max=32, step=1)
    nbRing: bpy.props.IntProperty(name="Number of rings", description="Number of sphere's rings", default=8, min=3,
                                  max=16, step=1)
    # Define the "cube" where the adaptatives shapes can be
    x_max: bpy.props.FloatProperty(name="x maximum", description="Maximum x coordinate where cameras are displayed", default=100, min=-1000, max=1000, step=1)
    x_min: bpy.props.FloatProperty(name="x minimum", description="Minimum x coordinate where cameras are displayed", default=-100, min=-1000, max=1000, step=1)
    y_max: bpy.props.FloatProperty(name="y maximum", description="Maximum y coordinate where cameras are displayed", default=100, min=-1000, max=1000, step=1)
    y_min: bpy.props.FloatProperty(name="y minimum", description="Minimum y coordinate where cameras are displayed", default=-100, min=-1000, max=1000, step=1)
    z_max: bpy.props.FloatProperty(name="z maximum", description="Maximum z coordinate where cameras are displayed", default=100, min=-1000, max=1000, step=1)
    z_min: bpy.props.FloatProperty(name="z minimum", description="Minimum z coordinate where cameras are displayed", default=-100, min=-1000, max=1000, step=1)
    
    focalLength: bpy.props.FloatProperty(name="Focal length", description="Focal length of all cameras of the cluster in millimeter)")


classes = [SetupOperator, SetupProperties]


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SetupProperties = bpy.props.PointerProperty(type=SetupProperties)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.SetupProperties


if __name__ == "__main__":
    register()

