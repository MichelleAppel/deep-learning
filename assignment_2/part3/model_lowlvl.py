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
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        self.W_gx = nn.ParameterList().to(device) # modulation gate input weight matrix
        self.W_gh = nn.ParameterList().to(device) # modulation gate hidden weight matrix

        self.W_ix = nn.ParameterList().to(device) # input gate input weight matrix
        self.W_ih = nn.ParameterList().to(device) # modulation gate hidden weight matrix

        self.W_fx = nn.ParameterList().to(device) # forget gate input weight matrix
        self.W_fh = nn.ParameterList().to(device) # forget gate hidden weight matrix

        self.W_ox = nn.ParameterList().to(device) # output gate input weight matrix
        self.W_oh = nn.ParameterList().to(device) # output gate hidden weight matrix

        self.W_px = nn.ParameterList().to(device) # prediction gate input weight matrix
        self.W_ph = nn.ParameterList().to(device) # prediction gate hidden weight matrix

        self.b_g = nn.ParameterList().to(device) # modulation bias
        self.b_i = nn.ParameterList().to(device) # input bias
        self.b_f = nn.ParameterList().to(device) # forget bias
        self.b_o = nn.ParameterList().to(device) # output bias
        self.b_p = nn.ParameterList().to(device) # prediction bias

        for layer in range(lstm_num_layers):
            self.W_gx.append(nn.Parameter(torch.randn((vocabulary_size, lstm_num_hidden)).to(device))) # modulation gate input weight matrix
            self.W_gh.append(nn.Parameter(torch.randn((lstm_num_hidden, lstm_num_hidden)).to(device))) # modulation gate hidden weight matrix

            self.W_ix.append(nn.Parameter(torch.randn((vocabulary_size, lstm_num_hidden)).to(device))) # input gate input weight matrix
            self.W_ih.append(nn.Parameter(torch.randn((lstm_num_hidden, lstm_num_hidden)).to(device))) # input gate hidden weight matrix

            self.W_fx.append(nn.Parameter(torch.randn((vocabulary_size, lstm_num_hidden)).to(device))) # forget gate input weight matrix
            self.W_fh.append(nn.Parameter(torch.randn((lstm_num_hidden, lstm_num_hidden)).to(device))) # forget gate hidden weight matrix
                        
            self.W_ox.append(nn.Parameter(torch.randn((vocabulary_size, lstm_num_hidden)).to(device))) # output gate input weight matrix
            self.W_oh.append(nn.Parameter(torch.randn((lstm_num_hidden, lstm_num_hidden)).to(device))) # output gate hidden weight matrix

            self.W_px.append(nn.Parameter(torch.randn((vocabulary_size, vocabulary_size)).to(device))) # prediction gate input weight matrix
            self.W_ph.append(nn.Parameter(torch.randn((lstm_num_hidden, vocabulary_size)).to(device))) # prediction gate hidden weight matrix

            self.b_g.append(nn.Parameter(torch.zeros(lstm_num_hidden).to(device))) # modulation bias
            self.b_i.append(nn.Parameter(torch.zeros(lstm_num_hidden).to(device))) # input bias
            self.b_f.append(nn.Parameter(torch.zeros(lstm_num_hidden).to(device))) # forget bias
            self.b_o.append(nn.Parameter(torch.zeros(lstm_num_hidden).to(device))) # output bias
            self.b_p.append(nn.Parameter(torch.zeros(vocabulary_size).to(device))) # prediction bias

        self.tanh = nn.Tanh() # tanh module
        self.sigmoid = nn.Sigmoid() # softmax module
        self.softmax = nn.Softmax(dim=1) # softmax module

    def forward(self, x):
        # Make one-hot vector out of input idx
        batch_size = x.shape[0]
        x_onehot = torch.FloatTensor(self.seq_length, batch_size, self.vocabulary_size).to(self.device)
        x_onehot.zero_()
        ones = torch.ones(torch.t(x)[0].shape).to(self.device)
        for t, x_t in enumerate(torch.t(x)):
            x_onehot[t].scatter_(1, x_t.reshape(-1,1), ones.reshape(-1,1))
        x = x_onehot

        for layer in range(self.lstm_num_layers): # loop through lstm layers
            print(layer)
            h = torch.zeros(batch_size, self.num_hidden).to(self.device) # last hidden state
            c = torch.zeros(batch_size, self.num_hidden).to(self.device) # last hidden cell state
            prediction = []

            for x_t in x: # loop through sequence
                g_t = self.tanh(x_t @ self.W_gx[layer] + h @ self.W_gh[layer] + self.b_g[layer]) # modulation (equation (4))
                i_t = self.sigmoid(x_t @ self.W_ix[layer] + h @ self.W_ih[layer] + self.b_i[layer]) # input (equation (5))
                f_t = self.sigmoid(x_t @ self.W_fx[layer] + h @ self.W_fh[layer] + self.b_f[layer]) # forget (equation (6))
                o_t = self.sigmoid(x_t @ self.W_ox[layer] + h @ self.W_oh[layer] + self.b_o[layer]) # output (equation (7))

                c_t = g_t * i_t + c * f_t # cell (equation (8))
                h_t = self.tanh(c_t) * o_t # hidden (equation (9))

                c = c_t # store last cell
                h = h_t # store last hidden state

                p_t = h_t @ self.W_ph[layer] + self.b_p[layer] # prediction (equation (10))
                # y_t = self.softmax(p_t) # softmax (equation (11))
                prediction.append(p_t)
            x = prediction # output is new input
        return prediction