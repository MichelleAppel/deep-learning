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

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()

        self.num_hidden = lstm_num_hidden
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size

        self.W_gx = nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden)) # modulation gate input weight matrix
        self.W_gh = nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden)) # modulation gate hidden weight matrix

        self.W_ix = nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden)) # input gate input weight matrix
        self.W_ih = nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden)) # input gate hidden weight matrix

        self.W_fx = nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden)) # forget gate input weight matrix
        self.W_fh = nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden)) # forget gate hidden weight matrix
                     
        self.W_ox = nn.Parameter(torch.randn(vocabulary_size, lstm_num_hidden )) # output gate input weight matrix
        self.W_oh = nn.Parameter(torch.randn(lstm_num_hidden, lstm_num_hidden )) # output gate hidden weight matrix

        self.W_px = nn.Parameter(torch.randn(vocabulary_size, vocabulary_size)) # prediction gate input weight matrix
        self.W_ph = nn.Parameter(torch.randn(lstm_num_hidden, vocabulary_size)) # prediction gate hidden weight matrix

        self.b_g = nn.Parameter(torch.zeros(lstm_num_hidden)) # modulation bias
        self.b_i = nn.Parameter(torch.zeros(lstm_num_hidden)) # input bias
        self.b_f = nn.Parameter(torch.zeros(lstm_num_hidden)) # forget bias
        self.b_o = nn.Parameter(torch.zeros(lstm_num_hidden)) # output bias
        self.b_p = nn.Parameter(torch.zeros(vocabulary_size)) # prediction bias

        self.tanh = nn.Tanh() # tanh module
        self.sigmoid = nn.Sigmoid() # softmax module
        self.softmax = nn.Softmax(dim=1) # softmax module

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden) # last hidden state
        c = torch.zeros(self.batch_size, self.num_hidden)
        prediction = []
        
        for x_t in torch.t(x): # loop through sequence
            x_tonehot = torch.FloatTensor(self.batch_size, self.vocabulary_size)
            x_tonehot.zero_()
            ones = torch.ones(x_t.shape)
            x_tonehot.scatter_(1,x_t.reshape(-1,1), ones.reshape(-1,1))

            x_t = x_tonehot

            g_t = self.tanh(x_t @ self.W_gx + h @ self.W_gh + self.b_g) # modulation (equation (4))
            i_t = self.sigmoid(x_t @ self.W_ix + h @ self.W_ih + self.b_i) # input (equation (5))
            f_t = self.sigmoid(x_t @ self.W_fx + h @ self.W_fh + self.b_f) # forget (equation (6))
            o_t = self.sigmoid(x_t @ self.W_ox + h @ self.W_oh + self.b_o) # output (equation (7))

            c_t = g_t * i_t + c * f_t # cell (equation (8))
            h_t = self.tanh(c_t) * o_t # hidden (equation (9))

            c = c_t # store last cell
            h = h_t # store last hidden state

            p_t = h_t @ self.W_ph + self.b_p # prediction (equation (10))
            y_t = self.softmax(p_t) # softmax (equation (11))
            prediction.append(y_t)
        return prediction
