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

import os
import time
from datetime import datetime
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

from tensorboardX import SummaryWriter

################################################################################

def train(config):

    if config.summary:
        # Tensorboard writer
        writer = SummaryWriter(
            os.path.join(config.summary_path, config.txt_file[config.txt_file.rfind('/')+1:], 'seq_length_' + str(config.seq_length)),
            filename_suffix='.seq_length_' + str(config.seq_length))

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length, newline_to_whitespace=config.newline_to_whitespace, rm_special=config.rm_special)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.learning_rate_step, gamma=config.learning_rate_decay) # Learning rate decay

    # dropout = nn.Dropout(p=1-config.dropout_keep_prob)
    total_step = 0

    for epoch in range(config.epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            # Counter
            total_step += 1
            step = total_step

            # Only for time measurement of step through network
            t1 = time.time()

            # Transform input to tensor
            batch_inputs = torch.t(torch.stack(batch_inputs)).to(device) 

            # Set gradients to zero
            optimizer.zero_grad() 

            # Start learning rate decay after certain amount of steps
            if step > config.learning_rate_decay_after:
                scheduler.step() 

            # Predict sequence
            prediction = model(batch_inputs) 

            # Calculate loss and accuracy
            loss = 0
            accuracy = 0
            for prediction_t, target_t in zip(prediction, batch_targets):
                target_t = target_t.to(device)
                loss += criterion(prediction_t, target_t) # Loss
                accuracy += float(torch.sum(prediction_t.argmax(dim=1)==target_t))/config.batch_size # Accuracy
            accuracy = accuracy/config.seq_length

            # Tensorboard summary
            if config.summary:
                writer.add_scalar('loss', loss, step)
                writer.add_scalar('accuracy', accuracy, step)

            # The optimization process
            loss.backward() # Perform backward pass
            optimizer.step() # Update weights

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)

            # Print progress
            if step % config.print_every == 0:
                print("[{}] Epoch {}, Train Step {:d}/{:04d}, LR = {}, Batch Size = {}, Examples/Sec = {:.2f}, "
                    "Accuracy = {:.2f}, Loss = {:.3f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M"), epoch, step,
                        int(config.train_steps), optimizer.param_groups[0]['lr'], config.batch_size, examples_per_second,
                        accuracy, loss
                ))            

            # Show some predictions every certain amount of steps
            if step % config.sample_every == 0:
                prediction_idx = torch.t(torch.stack([p_t.argmax(dim=1) for p_t in prediction]))
                batch_targets = torch.t(torch.stack(batch_targets))

                input_string = dataset.convert_to_string(batch_inputs[0])
                target_string = dataset.convert_to_string(batch_targets[0])
                predicted_string = dataset.convert_to_string(prediction_idx[0])

                print()
                print('INPUT: ', input_string)
                print('TARG:  ', target_string)
                print('PRED:  ', predicted_string)
                print()

                # And add them to the summary if indicated
                if config.summary:
                    writer.add_text('input', input_string, step)
                    writer.add_text('target', target_string, step)
                    writer.add_text('output', predicted_string, step)

            # Save the model every certain amount of steps
            if step % config.save_model_every == 0:
                model_path = os.path.join(config.save_model_dir, config.txt_file[config.txt_file.rfind('/')+1:])
                if not os.path.exists(model_path):
                     os.makedirs(model_path)
                torch.save(model, os.path.join(model_path, 'step_' + str(step) + '.pt'))

        if step > config.train_steps:
            break

    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, default='data/alice.txt', help="Path to a .txt file to train on")
    parser.add_argument('--newline_to_whitespace', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Replace newlines with whitespace in the txt file")
    parser.add_argument('--rm_special', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Replace newlines with whitespace in the txt file")
    parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')

    # It is not necessary to implement the following three params, but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_decay_after', type=int, default=1e5, help='Learning rate decay starts after number of steps')
    parser.add_argument('--learning_rate_step', type=int, default=2e3, help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help='Make summary', )
    parser.add_argument('--summary_path', type=str, default="./summaries/", help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5, help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--save_model_every', type=int, default=1e4, help="When to save the model")
    parser.add_argument('--save_model_dir', type=str, default='./model/', help="The directory to save the model in")

    config = parser.parse_args()

    # Train the model
    train(config)