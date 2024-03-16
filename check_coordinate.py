import torch as th
import numpy as np


batch_size = 1
maxt = 1000 # 1000 개의 1
w = np.ones([maxt])
p = w / np.sum(w)
#
indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
print(f'indices_np = {indices_np}')
indices = th.from_numpy(indices_np).long()
weights_np = 1 / (len(p) * p[indices_np]) # 1
weights = th.from_numpy(weights_np).float()
print(weights)