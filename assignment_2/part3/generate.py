import os
import time
import argparse

import torch
import torch.nn.functional as F

from dataset import TextDataset
import numpy as np

def generate(config):

    # The device
    device = torch.device(config.device)

    # Load the saved model
    model = torch.load(config.model_file) 

    # Load the dataset
    dataset = TextDataset(config.txt_file, model.seq_length, newline_to_whitespace=config.newline_to_whitespace, rm_special=config.rm_special)

    # The final output tensor
    final_output_idx = torch.LongTensor(config.generate_length).to(device) # Setup final idx tensor
    final_output_idx.zero_() # Set to zero
    final_output_idx[0].random_(0, dataset.vocab_size) # First char is random

    # Generate characters
    for idx in range(1, config.generate_length):
        if idx > model.seq_length:
            # Predictions with idx larger than sequence length
            pred_idx = -1
            seq_begin = idx - model.seq_length
            seq_end = idx
        else: 
            # First seq_len predictions
            pred_idx = idx - 1
            seq_begin = 0
            seq_end = model.seq_length

        # The predicted character
        prediction = model(final_output_idx[seq_begin:seq_end].reshape(1,-1))[pred_idx] # Tensor containing model output
        probs = F.softmax(prediction, dim=1) # Transform to probabilities

        if config.temperature_sampling:
            # Temperature sampling
            distribution = torch.distributions.Categorical(probs / config.temperature) # Make a distribution to sample from
            final_output_idx[idx] = distribution.sample().to(device)[0] # Sample with probability from the distribution
        else:
            # Greedy sampling
            final_output_idx[idx] = probs.argmax(dim=1) # Add to final prediction list

    output_string = dataset.convert_to_string(final_output_idx) # As string
    print(output_string)

    if config.save_output:
        output_path = os.path.join('output', config.txt_file[config.txt_file.rfind('/')+1:])
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        text_file = open(os.path.join(output_path, str(time.time())+'.txt' ) , "w")
        text_file.write(output_string)
        text_file.close()
    
if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_file', type=str, default='model/alice.txt/step_2000.pt', help="Path to the model")
    parser.add_argument('--txt_file', type=str, default='data/alice.txt', help="Path to a .txt file on which the model was trained")
    parser.add_argument('--newline_to_whitespace', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Replace newlines with whitespace in the txt file")
    parser.add_argument('--rm_special', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Replace newlines with whitespace in the txt file")
    
    parser.add_argument('--temperature_sampling', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Whether to perform temperature sampling")
    parser.add_argument('--temperature', type=float, default=0.5, help="Set the temperature for temperature sampling")

    parser.add_argument('--generate_length', type=int, default=1000, help="Amount of characters to generate")
    parser.add_argument('--save_output', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Save output string to txt file")

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    generate(config)