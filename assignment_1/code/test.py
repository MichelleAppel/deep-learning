from mlp_numpy import *

mlp = MLP(3,[4,5,4],3)

print(mlp.forward([10,1,-4]))
