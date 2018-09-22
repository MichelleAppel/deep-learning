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

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cpu'):
        super(VanillaRNN, self).__init__()
        self.num_hidden = num_hidden

        self.W_hx = nn.Parameter(torch.ones(input_dim , num_hidden )) # input-to-hidden weight matrix
        self.W_hh = nn.Parameter(torch.ones(num_hidden, num_hidden )) # hidden-to-hidden weight matrix
        self.W_ph = nn.Parameter(torch.ones(num_hidden, num_classes)) # hidden-to-output weight matrix

        self.b_h = torch.nn.Parameter(torch.zeros(num_hidden )) # hidden bias
        self.b_p = torch.nn.Parameter(torch.zeros(num_classes)) # output bias

        self.tanh = nn.Tanh()       # tanh module
        self.softmax = nn.Softmax(dim=0) # softmax module

    def forward(self, x):
        h = [torch.zeros(self.num_hidden, self.num_hidden)] # list of hidden states
        
        for x_t in x: # loop through sequence
            x_t = x_t.reshape((-1,)) # reshape input at timestep t
            h_t = self.tanh( x_t @ self.W_hx + h[-1] @ self.W_hh + self.b_h) # hidden state at timestep t
            h.append(h_t) # store in list

            p_t = h_t @ self.W_ph + self.b_p # output value

        y_p = self.softmax(torch.sum(p_t, dim=0)) # softmax TODO: sum??
        return y_p
