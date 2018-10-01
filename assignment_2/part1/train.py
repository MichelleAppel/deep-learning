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

import argparse
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM

from tensorboardX import SummaryWriter

################################################################################

def train(config):

    assert config.model_type in ('RNN', 'LSTM')

    # Make summary
    writer = SummaryWriter('./summaries/'+config.model_type+'/input_length_' + str(config.input_length), 
        filename_suffix='.input_length_' + str(config.input_length))

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the model that we are going to use
    if config.model_type == 'RNN':
        model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device=device)
    elif config.model_type == 'LSTM':
        model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size, device=device)

    # Initialize the dataset and data loader (note the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Only for time measurement of step through network
        t1 = time.time()

        # Set gradients to zero
        optimizer.zero_grad()

        # Predict the last number of the palindrome
        prediction = model(batch_inputs)

        ############################################################################
        # QUESTION: what happens here and why?
        # This function clips the norm of the gradients by scaling the gradients down 
        # in order to reduce the norm to an acceptable level such that the update
        # doesn't get too large. 
        ############################################################################
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        ############################################################################

        # Move to GPU
        batch_targets = batch_targets.to(device)

        # Calculate loss and accuracy
        loss = criterion(prediction, batch_targets)
        accuracy = float(torch.sum(prediction.argmax(dim=1)==batch_targets))/config.batch_size

        # Tensorboard summary
        writer.add_scalar('loss', loss, step)
        writer.add_scalar('accuracy', accuracy, step)

        # Update of the network
        loss.backward() # Perform backward pass
        optimizer.step() # Update weights

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        # Print progress
        if step % 10 == 0:
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
            ))

        if step == config.train_steps:
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()
    
    # Train the model
    train(config)