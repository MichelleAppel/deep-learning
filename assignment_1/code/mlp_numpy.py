"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
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
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes

    self.layers = [] # List of layers

    if len(n_hidden) == 0: # If no hidden layers
      self.layers.append(LinearModule(n_inputs, n_classes)) # Linear module to softmax layer
    else:
      for i in range(len(n_hidden)): # Append hidden layers
        if i == 0:
          n_in = n_inputs
        else:
          n_in = self.n_hidden[i-1]
        self.layers.append(LinearModule(n_in, self.n_hidden[i]))
        self.layers.append(ReLUModule())
      self.layers.append((LinearModule(self.n_hidden[-1], self.n_classes))) # Final linear module
    self.layers.append(SoftMaxModule()) # Softmax layer for prediction

    # for layer in self.layers:
      # print(layer)
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
    for layer in self.layers: # Loop through layers
      # print(out.shape)
      out = layer.forward(out) # Forward pass with previous output as new input
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def loss(self, out, labels):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      out: the final output of the network
      labels: the ground truth
    Returns:
      loss: the loss of the prediction
      dloss: the gradients of the loss
    """

    loss = self.crossentropy_layer.forward(out, labels)
    dloss = self.crossentropy_layer.backward(out, labels)

    return loss, dloss

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    Done:
    Implement backward pass of the network.
    """
    
    ######################
    # PUT YOUR CODE HERE  #
    #######################
    for layer in reversed(self.layers): # Loop trough layers in reversed order
      dout = layer.backward(dout) # Backward pass with previous gradients as input
    #######################
    # END OF YOUR CODE    #
    #######################

    return dout