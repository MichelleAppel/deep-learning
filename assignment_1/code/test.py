from mlp_pytorch import *
import numpy as np
import torch

mlp = MLP(3,[3,3],3)

out = mlp.forward(torch.rand(3))
print(out)