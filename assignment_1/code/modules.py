"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    Done:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.in_features = in_features # Store the amount of input features
    self.out_features = out_features # Store the amount of output features

    mu, sigma = 0, 0.0001 # Mean and standard deviation

    weight = np.random.normal(mu, sigma, (in_features, out_features)) # Initialize random weights
    bias = np.zeros((1, out_features)) # Initialize bias with 0

    grads = np.zeros((in_features, out_features)) # Initialize gradients with 0

    self.params = {'weight': weight, 'bias': bias} # Store the parameters
    self.grads = {'weight': grads, 'bias': bias} # Store the gradients

    self.x = None # Future storage for input
    self.out = None # Future storage for output
    #######################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Done:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    out = np.dot(x, self.params['weight']) + self.params['bias'] # Output = Weights * input + bias

    self.x = x # Store input
    self.out = out # Store output
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    Done:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dx = self.params['weight'] # Gradient with respect to input is weights
    dx = dx * dout # Multiply elementwise with gradients of previous module

    self.grads['weight'] = self.x # Gradient of weight is input
    self.grads['bias'] = np.ones((1, self.out_features)) # Gradient of bias equals one
    #######################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class ReLUModule(object):
  """
  ReLU activation module.
  """
  def __init__(self):
    self.x = None # Future storage for input
    self.out = None # Future storage for output

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module

    # Done:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x # Store input

    out = x.copy() # Copy input
    out[np.less(out, 0)] = 0 # ReLU: output = 0 if input is negative

    self.out = out # Store output
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    # Done:
    Implement backward pass of the module.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.zeros(np.shape(self.x)) # Gradient is 0 when input < 0
    dx[np.greater_equal(self.x, 0)] = 1 # Gradient is 1 when input >= 0
    dx = dx * dout # Elementwise multiplication with previous gradients
    #######################
    # END OF YOUR CODE    #
    #######################    

    return dx

class SoftMaxModule(object):
  """
  Softmax activation module.
  """
  def __init__(self):
    self.x = None # Future storage for input
    self.out = None # Future storage for output

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    Done:
    Implement forward pass of the module. 
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x # Store input

    b = x.max() 
    y = np.exp(x - b) # Exp-normalize trick to avoid numerical overflow
    out = y / y.sum() # Softmax

    self.out = out # Store output
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dx = self.x - self.x**2 # if i==j
    # dx_ij = -self.x*self.x # if i!=j TODO

    dx = dx * dout # Elementwise multiplication with gradients of previous module
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss
    
    Done:
    Implement forward pass of the module. 
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    out = -np.sum(y*np.log(x)) # Cross entropy
    #######################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.

    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.
    
    Done:
    Implement backward pass of the module.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    dx = y/x # Derivative of cross entropy module
    #######################
    # END OF YOUR CODE    #
    #######################

    return dx
