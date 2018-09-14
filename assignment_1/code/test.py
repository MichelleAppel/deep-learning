# from mlp_numpy import MLP

# mlp = MLP(3, (3,), 2)
# print(mlp.forward([2,1,2]))

from modules import *

softmax = SoftMaxModule()
out = softmax.forward(np.array([8,0.1,0.1]))
print(out)

crossentropy = CrossEntropyModule()
loss = crossentropy.forward(out, [1,0,0])

out = softmax.backward(loss)
print(out)