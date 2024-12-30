# SED Creator
***
SEDCreator is a Blender add-on which allows the user to add clusters of cameras in a Blender Scene.

## Installation

+ Compress the _SEDCreator_ folder in _zip_ format.
+ Open Blender, go to _Edit_, then _Preferences_ (or _Ctrl_ '_,_' shortcut). The _Preferences_ window will open.
+ Go to _Add-ons_, _Install_ and choose the _SEDCreator.zip_ you previously created.
+ Type _SEDcreator_ in the research bar and you should see _Generic: SEDcreator_. Check this box.
+ Click on the hamburger menu and click on _Save preferences_. You are done! You can close the _Preferences_ window.

## How to use

- Press '_n_' to make the _Sidebar_ of the _View Menu_ appear. You should now see the new _SEDcreator_ tab.

### Scene Management
You can now add a cluster of cameras in your Blender Scene: 
- Place one or multiple _Empties_ in the scene and select it or them, then click on _Set Project Setup_.  
**Note**: The setup you choose will only affect the _Empties_ that have been selected. At least one _Empty_ must be selected for the setup to take place. _Empties_ that are not selected during this process, will not be affected whatsoever.
- There are several types of camera clusters with various options to choose from (radius, focal length of cameras, orientation of cameras). You can try them all to find the one(s) that best fit your project's needs!
>[!TIP]
> The _Focus_ option (in Camera orientation) is a bit tricky: you must first select the object you want your cameras to "look at", then click on _Set cluster focus_.

### Delimitations
The _Delimitations_ are for the _Adaptative Icosahedron_ and _Adaptative UV Sphere_ only.  
This is to limit the area in which cameras can be created (i.e. if you place an _Empty_, and some cameras of the future cluster should later appear outside of the _Delimitations_, these cameras will not be created.

### Camera setup
Click on the _Setup cameras_ button to create or modify the camera cluster around the selected _Empty_.

### Render Management
When you are ready to render, follow these steps:  
- Check the boxes according to the images you want to render (_Albedo_, _Depth_, etc.).  
**Note**: Beauty is rendered by default (i.e. you cannot choose to not have it be rendered).
- Name the directory you want your images to be rendered to.  
**Note**: The directory will be created in the same place as your Blender file.
- _First frame_ and _Last frame_ are the range of cameras you want to render.  
**Note**: One camera is equal to one frame (with the corresponding numbers).  
Let's say you want to render the images from _Camera_11_ to _Camera_32_ only, you should enter _First frame = 11_ and _Last frame = 32_.
- Click on _Start Render_.

## Two versions
There are two versions of this add-on, one on branch _main_ and the second one on branch _alternative_version_. The only difference between these two versions, concerns the rendering process:  
- In the first version, if you hide cameras in the viewport, they are still rendered (and numbered) as if they were in the scene.
- Meanwhile in the second version, the hidden cameras are completely ignored during the rendering process. If you cannot see the cameras, they do not exist.
