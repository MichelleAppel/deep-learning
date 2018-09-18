"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

INPUT_SIZE_DEFAULT = 3*32*32
OUTPUT_SIZE_DEFAULT = 10

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
	
	TODO:
	Implement accuracy computation.
	"""

	########################
	# PUT YOUR CODE HERE	#
	#######################
	pred = torch.max(predictions, 1)[1].long()
	accuracy = np.count_nonzero((pred==targets).data.numpy()) / float(np.shape(predictions)[0])
	########################
	# END OF YOUR CODE	#
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
	torch.manual_seed(42)

	## Prepare all functions
	# Get number of units in each hidden layer specified in the string such as 100,100
	if FLAGS.dnn_hidden_units:
		dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
		dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
	else:
		dnn_hidden_units = []

	########################
	# PUT YOUR CODE HERE	#
	#######################
	
	# read data
	cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
	
	def read_data_batch(mode='train', size=BATCH_SIZE_DEFAULT):
		inputs, labels = cifar10[mode].next_batch(size)
		inputs = Variable(torch.from_numpy(inputs), requires_grad=True)
		inputs = inputs.contiguous().view(-1, INPUT_SIZE_DEFAULT)
		labels = Variable(torch.from_numpy(labels).long())		
		labels = torch.max(labels, 1)[1]
		return inputs, labels
		
	def evaluate(mode='train'):
		inputs, labels = read_data_batch(mode, 10)
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		acc = accuracy(outputs, labels)
		return loss, acc
	
	# init network, criterion, optimizer
	net = MLP(INPUT_SIZE_DEFAULT, dnn_hidden_units, OUTPUT_SIZE_DEFAULT) 
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE_DEFAULT)
	
	# train loop
	train_acc = []
	test_acc = []
	train_loss = []
	test_loss = []
	for step in range(MAX_STEPS_DEFAULT):
		# get the input images and labels in correct format
		inputs, labels = read_data_batch()
		optimizer.zero_grad() # clear the previous gradients
		outputs = net(inputs) # forward pass
		loss = criterion(outputs, labels) # calculate loss		
		loss.backward() # compute new gradients
		optimizer.step() # update weights
		
		# evaluate on test set
		if step > 0 and step % 99 == 0:
			train_l, train_a = evaluate('train')
			test_l, test_a = evaluate('test')
			train_acc.append(train_a)
			test_acc.append(test_a)
			train_loss.append(train_l)
			test_loss.append(test_l)
			print("Train loss", np.round(train_l.data[0], 2), 
				"   Train acc", np.round(train_a, 2),
				"   Test loss", np.round(test_l.data[0], 2),
				"   Test acc", np.round(test_a, 2))

	print('Finished training :-)')
	########################
	# END OF YOUR CODE	#
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