import torch
import numpy as np

# get random 3 random number
shape = 3
first_coords = np.random.randint(0, 32+1, shape) + np.random.randint(0, 64+32+1, shape)
index = tuple([slice(None), *(slice(f, f+128) for f in first_coords      )])
print(index)
#for f in first_coords :
#    s = *slice(f, f + 128)
#    print(s)