from mlp_numpy import *
from modules import *

mlp = MLP(3,[2,2],3)

out = mlp.forward([10,1,-4])
loss = mlp.loss(out, [1,1,0])
dce = np.array(loss[1])
dout = mlp.backward(loss[1])