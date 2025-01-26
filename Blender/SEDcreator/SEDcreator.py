# Imports
import bpy

# Panel
class sedPanel(bpy.types.Panel):
    bl_idname = 'SEDCREATOR_PT_sedcreator'
    bl_label = 'SEDcreator Panel'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "SEDcreator"

    def draw(self, context):
        layout = self.layout
        setupProp = context.scene.SetupProperties
        domeShape = setupProp.domeShape

        pan_col1 = layout.column()
        pan_col1.label(text="Scene Management")

        row = pan_col1.row()
        row.prop(setupProp, 'domeShape')
        row = pan_col1.row()
        row.prop(setupProp, 'orientationCameras')
        if setupProp.orientationCameras == 'F':
            row = pan_col1.row()
            row.operator('object.sed_setfocus')
        row = pan_col1.row()
        row.prop(setupProp, 'clusterRadius')
        row = pan_col1.row()
        row.prop(setupProp, 'focalLength')
        if domeShape == "I" or domeShape == "SI" or domeShape == "AI":
            row = pan_col1.row()
            row.prop(setupProp, "nbSubdiv")
        if domeShape == "U" or domeShape == "SS" or domeShape == "AS":
            row = pan_col1.row()
            row.prop(setupProp, "nbSegment")
            row = pan_col1.row()
            row.prop(setupProp, "nbRing")
        row = pan_col1.row()
        layout.separator()

        pan_col2 = layout.column()
        pan_col2.label(text="Delimitations")

        row2 = pan_col2.row()
        row2.prop(setupProp, 'x_min')
        row2 = pan_col2.row()
        row2.prop(setupProp, 'x_max')
        row2 = pan_col2.row()
        row2.prop(setupProp, 'y_min')
        row2 = pan_col2.row()
        row2.prop(setupProp, 'y_max')
        row2 = pan_col2.row()
        row2.prop(setupProp, 'z_min')
        row2 = pan_col2.row()
        row2.prop(setupProp, 'z_max')
        row2 = pan_col2.row()
        layout.separator()

        pan_col3 = layout.column()
        pan_col3.label(text="Setup Cameras")

        row3 = pan_col3.row()
        row3.operator('object.sed_setup')

        if context.scene.RenderProperties.renderReady:

            layout.separator()

            pan_col4 = layout.column()
            pan_col4.label(text="Render Management")

            renderProp = context.scene.RenderProperties
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_albedo")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_depth")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_normal")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_id")
            row4 = pan_col4.row()
            row4.enabled = False
            row4.prop(renderProp, "bool_beauty")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_transmission")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_roughness")
            row4 = pan_col4.row()
            row4.prop(renderProp, "bool_curvature")
            row4 = pan_col4.row()
            row4.prop(renderProp, "exportFolder")
            row4 = pan_col4.row()
            row4.prop(renderProp, "start")
            row4 = pan_col4.row()
            row4.prop(renderProp, "end")
            row4 = pan_col4.row()
            row4.operator('object.sed_render')
            row4 = pan_col4.row()
            row4.operator('object.sed_save_cameras_attributes')
        else:
            row = layout.row()
            row.label(text="Render not ready")


classes = [sedPanel]


def register():
    for c in classes:
        bpy.utils.register_class(c)

def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
