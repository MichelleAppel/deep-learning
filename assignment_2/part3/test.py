from dataset import TextDataset
from model import TextGenerationModel
import numpy as np
import torch

dataset = TextDataset('data/alice.txt', 6)

item = dataset.__getitem__('ieks')
inputs, targets = item
# print(dataset.convert_to_string(item[0]))
# print(dataset.convert_to_string(item[1]))

model = TextGenerationModel(1, 6, dataset.vocab_size)
output = model.forward(torch.from_numpy(np.array(inputs)).reshape(1,-1))
print(output)