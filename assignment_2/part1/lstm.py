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

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, num_classes, batch_size, device='cuda:0'):
        super(LSTM, self).__init__()

        self.device = device

        self.num_hidden = num_hidden
        self.batch_size = batch_size

        self.W_gx = nn.Parameter(torch.randn(input_dim, num_hidden).to(self.device))  # modulation gate input weight matrix
        self.W_gh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(self.device)) # modulation gate hidden weight matrix

        self.W_ix = nn.Parameter(torch.randn(input_dim, num_hidden).to(self.device))  # input gate input weight matrix
        self.W_ih = nn.Parameter(torch.randn(num_hidden, num_hidden).to(self.device)) # input gate hidden weight matrix

        self.W_fx = nn.Parameter(torch.randn(input_dim, num_hidden).to(self.device))  # forget gate input weight matrix
        self.W_fh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(self.device)) # forget gate hidden weight matrix
                     
        self.W_ox = nn.Parameter(torch.randn(input_dim, num_hidden).to(self.device))  # output gate input weight matrix
        self.W_oh = nn.Parameter(torch.randn(num_hidden, num_hidden).to(self.device)) # output gate hidden weight matrix

        self.W_px = nn.Parameter(torch.randn(input_dim, num_classes).to(self.device))  # prediction gate input weight matrix
        self.W_ph = nn.Parameter(torch.randn(num_hidden, num_classes).to(self.device)) # prediction gate hidden weight matrix

        self.b_g = nn.Parameter(torch.zeros(num_hidden ).to(self.device)) # modulation bias
        self.b_i = nn.Parameter(torch.zeros(num_hidden ).to(self.device)) # input bias
        self.b_f = nn.Parameter(torch.zeros(num_hidden ).to(self.device)) # forget bias
        self.b_o = nn.Parameter(torch.zeros(num_hidden ).to(self.device)) # output bias
        self.b_p = nn.Parameter(torch.zeros(num_classes).to(self.device)) # prediction bias

        self.tanh = nn.Tanh().to(self.device) # tanh module
        self.sigmoid = nn.Sigmoid().to(self.device) # softmax module
        self.softmax = nn.Softmax(dim=1).to(self.device) # softmax module

    def forward(self, x):
        h = torch.zeros(self.batch_size, self.num_hidden).to(self.device) # last hidden state
        c = torch.zeros(self.batch_size, self.num_hidden).to(self.device) # last cell state

        for x_t in torch.t(x).to(self.device): # loop through sequence
            x_t = x_t.reshape(-1,1) # reshape input at timestep t
            g_t = self.tanh(x_t @ self.W_gx + h @ self.W_gh + self.b_g) # modulation (equation (4))
            i_t = self.sigmoid(x_t @ self.W_ix + h @ self.W_ih + self.b_i) # input (equation (5))
            f_t = self.sigmoid(x_t @ self.W_fx + h @ self.W_fh + self.b_f) # forget (equation (6))
            o_t = self.sigmoid(x_t @ self.W_ox + h @ self.W_oh + self.b_o) # output (equation (7))

            c_t = g_t * i_t + c * f_t # cell (equation (8))
            h_t = self.tanh(c_t) * o_t # hidden (equation (9))

            c = c_t # store last cell
            h = h_t # store last hidden state

        p_t = h_t @ self.W_ph + self.b_p # prediction (equation (10))
        # y_t = self.softmax(p_t) # softmax (equation (11)) already included in cross entropy loss apparently 
        return p_t