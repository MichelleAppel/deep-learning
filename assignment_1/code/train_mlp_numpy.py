"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import torch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NUM_CLASSES = 10

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  Done:
  Implement accuracy computation.
  """

  #######################
  # PUT YOUR CODE HERE  #
  #######################
  accuracy = 1-0.5*np.sum(~np.equal(predictions, targets))/np.shape(predictions)[0]
  #######################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  #######################
  # PUT YOUR CODE HERE  #
  #######################
  X_train_raw, Y_train_raw, X_test_raw, Y_test_raw = cifar10_utils.load_cifar10(DATA_DIR_DEFAULT)
  X_train, Y_train, X_test, Y_test = cifar10_utils.preprocess_cifar10_data(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw) # Load and preprocess dataset

  mlp = MLP(np.size(X_train[0]), dnn_hidden_units, NUM_CLASSES) # Initialize MLP
  print(X_train[0].size, dnn_hidden_units, NUM_CLASSES)

  for i in range(BATCH_SIZE_DEFAULT % np.shape(X_train)[0]): # Loop trough batches
    batch_input = X_train[i*BATCH_SIZE_DEFAULT:i*BATCH_SIZE_DEFAULT+BATCH_SIZE_DEFAULT] # Get batch input
    batch_input = np.reshape(batch_input, (BATCH_SIZE_DEFAULT, np.size(batch_input[0]))) # Reshape
    batch_output = Y_train[i*BATCH_SIZE_DEFAULT:i*BATCH_SIZE_DEFAULT+BATCH_SIZE_DEFAULT] # Get batch output
    # batch_output = cifar10_utils.dense_to_one_hot(batch_output, NUM_CLASSES) # Make one-hot vector

    prediction = mlp.forward(batch_input) # The predicted labels
    print(prediction)
    break
    
    loss, dloss = mlp.loss(prediction, batch_output) # Loss
    backward = mlp.backward(dloss) # Perform backward pass

    for layer in mlp.layers:
      if layer.__class__.__name__ == 'LinearModule':
        layer.params['weight'] -= LEARNING_RATE_DEFAULT*layer.grads['weight'] # Update weights
        layer.params['bias'] -= LEARNING_RATE_DEFAULT*layer.grads['bias'] # Update bias

  #######################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()