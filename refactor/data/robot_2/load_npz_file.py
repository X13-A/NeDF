import numpy as np

data = np.load('./unri/cameras_attributes.npz')

#je print juste les tableaux la, mais c'est juste pour avoir les noms des tableaux correspondants
print(data['cameras_angle'])
print(data['cameras_locations'])
