import torch

dim = 3
basic_grid = 3 * torch.linspace(-1, 1, 256)
mesh_grid = torch.meshgrid(basic_grid, indexing='ij')
coord_cache = torch.stack(mesh_grid, dim=0) # 1,256
image = torch.randn((5,256,256,256))
image = torch.cat([image, coord_cache], dim=0) # [5,256,256,256] -> [2,5,256,256,256]