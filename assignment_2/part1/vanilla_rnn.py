################################################################################
# MIT License
#
# Copyright (c) 2018
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

import torch
import torch.nn as nn

################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cuda:0'):
        super(VanillaRNN, self).__init__()

        self.device = torch.device(device)

        self.num_hidden = num_hidden
        self.batch_size = batch_size

        self.W_hx = nn.Parameter(torch.randn(input_dim , num_hidden ).to(self.device)) # input-to-hidden weight matrix
        self.W_hh = nn.Parameter(torch.randn(num_hidden, num_hidden ).to(self.device)) # hidden-to-hidden weight matrix
        self.W_ph = nn.Parameter(torch.randn(num_hidden, num_classes).to(self.device)) # hidden-to-output weight matrix

        self.b_h = nn.Parameter(torch.zeros(num_hidden ).to(self.device)) # hidden bias
        self.b_p = nn.Parameter(torch.zeros(num_classes).to(self.device)) # output bias

        self.tanh = nn.Tanh().to(self.device)            # tanh module
        self.softmax = nn.Softmax(dim=1).to(self.device) # softmax module

        print('ieks')

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device) # last hidden state

        for x_t in torch.t(x).to(self.device): # loop through sequence
            x_t = x_t.reshape(-1,1) # reshape input at timestep t
            h_t = self.tanh(x_t @ self.W_hx + h @ self.W_hh + self.b_h) # hidden state at timestep t (equation (1))
            h = h_t # store last hidden state

            p_t = h_t @ self.W_ph + self.b_p # output value (equation (2))

        # y_p = self.softmax(p_t) # softmax, already included in cross entropy loss apparently 
        return p_t
