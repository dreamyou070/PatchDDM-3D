import torch

dim = 3
axis_grid = torch.linspace(-1, 1, 256)
total_axis_grid = dim * [axis_grid,]
# repeat three time
meshgrid = torch.meshgrid(total_axis_grid, indexing='ij') # 256,256,256
print(meshgrid[0].shape)
print(meshgrid[1].shape)
print(meshgrid[2].shape)
coord_cache = torch.stack(meshgrid,
                          dim=0)

image = torch.randn(4,256,256,256)
image = torch.cat([image, coord_cache], dim=0) # [4,256,256,256] -> [2,5,256,256,256]
print(f'image = {image.shape}')