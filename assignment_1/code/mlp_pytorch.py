"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
    
    Done:
    Implement initialization of the network.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.linear_layers = []

    if len(n_hidden) == 0: 
      self.linear_layers.append(nn.Linear(n_inputs, n_classes))
    else:
      for i in range(len(n_hidden)):
        if i == 0:
          n_in = n_inputs
        else:
          n_in = n_hidden[i]
        self.linear_layers.append(nn.Linear(n_in, n_hidden[i]))
      self.linear_layers.append(nn.Linear(n_hidden[-1], n_classes))

    #######################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    Done:
    Implement forward pass of the network.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    out = x
    for linear_layer in self.linear_layers[:-1]:
      out = F.relu(linear_layer.forward(out))
    out = F.softmax(self.linear_layers[-1].forward(out), dim=0)
    #######################
    # END OF YOUR CODE    #
    #######################

    return out
