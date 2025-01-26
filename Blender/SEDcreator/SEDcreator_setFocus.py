import bpy

class SetFocus(bpy.types.Operator):
    bl_idname = "object.sed_setfocus"
    bl_label = "Set cluster focus"
    bl_description = "Set selected clusters on a specific object"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        selected_objects = context.selected_objects
        if len(selected_objects) == 0: 
            self.report({'ERROR'}, "You have no selected objects, please select one object")
        elif len(selected_objects) > 1:
            self.report({'ERROR'}, "You have selected multiple objects, please only select one")
        else:
            context.scene.SetFocusProperties.focus_object = selected_objects[0].name
        return {'FINISHED'}


class SetFocusProperties(bpy.types.PropertyGroup):
    focus_object: bpy.props.StringProperty(name="Focus object", default="Not focus", description="Object to be focus on") 

classes = [SetFocus, SetFocusProperties]

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.SetFocusProperties = bpy.props.PointerProperty(type=SetFocusProperties)


def unregister():
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
    del bpy.types.Scene.SetFocusProperties


if __name__ == "__main__":
    register()
