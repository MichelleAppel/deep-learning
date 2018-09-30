# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

import numpy as np

# torch.manual_seed(42)

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.device = device

        self.layers = nn.ModuleList().to(device) # the list that is going to contain the lstm layers
        self.lstm = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden, num_layers=lstm_num_layers, dropout=0).to(device)
        self.linear = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size, bias=True).to(device)

    def forward(self, x):
        # Make one-hot vector out of input idx
        batch_size = x.shape[0]
        x_onehot = torch.FloatTensor(self.seq_length, batch_size, self.vocabulary_size).to(self.device)
        x_onehot.zero_()
        ones = torch.ones(torch.t(x)[0].shape).to(self.device)
        for t, x_t in enumerate(torch.t(x)):
            x_onehot[t].scatter_(1, x_t.reshape(-1,1), ones.reshape(-1,1))
        x = x_onehot

        output, _ = self.lstm(x)
        output = self.linear(output)
        return output