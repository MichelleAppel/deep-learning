from dataset import TextDataset
from model import TextGenerationModel
import numpy as np
import torch

dataset = TextDataset('data/alice.txt', 6)

inputs1, targets1 = dataset.__getitem__('ieks')
inputs2, targets2 = dataset.__getitem__('ieks')

inputs = [torch.from_numpy(np.array(inputs1)).reshape(1,-1), torch.from_numpy(np.array(inputs2)).reshape(1,-1)]

# print(dataset.convert_to_string(item[0]))
# print(dataset.convert_to_string(item[1]))

model = TextGenerationModel(2, 6, dataset.vocab_size)
output = model.forward(inputs)
# print(output)