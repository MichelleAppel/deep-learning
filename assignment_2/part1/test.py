from vanilla_rnn import VanillaRNN
from dataset import PalindromeDataset
import torch

seq_len = 5

dataset = PalindromeDataset(seq_len)
x = torch.from_numpy(dataset.__getitem__()[0])
print(x)

rnn = VanillaRNN(seq_len, 1, 2, 10, 1)

y = rnn.forward(x)
print(y)