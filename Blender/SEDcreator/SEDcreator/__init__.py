import sys
import importlib

#---------- Plugins Information ----------#
bl_info = {
    "name": "SEDcreator",
    "authors": "Maxime Verna, Sara Lafleur, Léa Touchard, Alexandre Miralles",
    "description": "Add-on to create clusters of cameras to create set of image of a scene",
    "blender": (4, 0, 0),
    "location": "",
    "warning": "",
    "category": "Generic"
}

# Register
modulesNames = ['SEDcreator', 'SEDcreator_createCluster', 'SEDcreator_utils', 'SEDcreator_setupClasses', 'SEDcreator_renderClasses', 'SEDcreator_render', 'SEDcreator_launchRender', 'SEDcreator_renderRoughness', 'SEDcreator_renderCurvature', 'SEDcreator_prepareRender', 'SEDcreator_setFocus']

modulesFullNames = {}
for currentModuleName in modulesNames:
    modulesFullNames[currentModuleName] = (
        '{}.{}'.format(__name__, currentModuleName))

for currentModuleFullName in modulesFullNames.values():
    if currentModuleFullName in sys.modules:
        importlib.reload(sys.modules[currentModuleFullName])
    else:
        globals()[currentModuleFullName] = importlib.import_module(
            currentModuleFullName)
        setattr(globals()[currentModuleFullName],
                'modulesNames', modulesFullNames)


def register():
    for currentModuleName in modulesFullNames.values():
        if currentModuleName in sys.modules:
            if hasattr(sys.modules[currentModuleName], 'register'):
                sys.modules[currentModuleName].register()


def unregister():
    for currentModuleName in modulesFullNames.values():
        if currentModuleName in sys.modules:
            if hasattr(sys.modules[currentModuleName], 'unregister'):
                sys.modules[currentModuleName].unregister()


if __name__ == "__main__":
    register()
