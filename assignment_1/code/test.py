# from mlp_pytorch import MLP
# import numpy as np
# import torch

# mlp = MLP(3,[4,6], 2)

# input = torch.from_numpy(np.array([1,2,3]))

# input = input.float()
# out = mlp.forward(input)
# print(out)

from modules import *

lin = LinearModule(3,2)

print(lin.forward([-1,3,1]))


print(lin.backward([[-1,3,3],[4,1,2]]))