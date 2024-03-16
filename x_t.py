import torch

image_channels = 7
x_t = torch.randn(1,8,128,128,128)
res = torch.randn(1,1,128,128,128)
# [1,1,128,128,128]
x_t[:, image_channels:, ...] = res

