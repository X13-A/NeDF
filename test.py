import glm
import torch

point = torch.tensor([0.0, 0.0, -5, 1.0])
glm_projection_matrix = glm.perspective(glm.radians(30), 1920 / 1080, 0.01, 500)
projection_matrix = torch.tensor(glm_projection_matrix.to_list(), dtype=torch.float32).T
            
clip_space_pos = torch.matmul(projection_matrix, point)
ndc_pos = clip_space_pos / clip_space_pos[3]
print(clip_space_pos)
print(ndc_pos)
