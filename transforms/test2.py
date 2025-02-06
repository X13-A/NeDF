import torch
from estimate_distance import estimate_distances
from dataset import load_dataset
from settings import *
import matplotlib.pyplot as plt

device = 'cuda:0'

dataset = load_dataset()
index = 0

depth_map = dataset[index][DEPTH_MAP_ENTRY]
plt.imshow(depth_map)
plt.show()
