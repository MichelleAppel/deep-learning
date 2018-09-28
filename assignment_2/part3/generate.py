import os
import argparse

import torch

from dataset import TextDataset

def generate(config):

    device = torch.device(config.device)

    model = torch.load(config.model_file) # Load the model

    dataset = TextDataset(config.txt_file, model.seq_length, newline_to_whitespace=config.newline_to_whitespace) # Get vocabulary size

    final_output_idx = torch.LongTensor(config.generate_length).to(device) # Setup final idx tensor
    final_output_idx.zero_() # Set to zero
    final_output_idx[0].random_(0, dataset.vocab_size) # First char is random

    # Generate first seq_length characters
    for idx in range(1, config.generate_length):
        if idx >= model.seq_length: # Rest of predictions
            pred_idx = -1
            seq_begin = idx - model.seq_length
            seq_end = idx
        else: # First 8 predictions
            pred_idx = idx - 1
            seq_begin = 0
            seq_end = model.seq_length

        print(final_output_idx[seq_begin:seq_end].reshape(1,-1))
        prediction = model(final_output_idx[seq_begin:seq_end].reshape(1,-1))[pred_idx] # The predicted character
        final_output_idx[idx] = prediction.argmax(dim=1) # Add to final prediction list

    print(dataset.convert_to_string(final_output_idx)) # Show as string

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, default='data/abc.txt', help="Path to a .txt file on which the model was trained")
    parser.add_argument('--newline_to_whitespace', type=lambda s: s.lower() in ['true', 't', 'yes', '1'], default=True, help="Replace newlines with whitespace in the txt file")
    
    parser.add_argument('--model_file', type=str, default='./model/test.txt/step_500.pt', help="Path to the model")

    parser.add_argument('--generate_length', type=int, default=10, help="Amount of characters to generate")

    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    # Train the model
    generate(config)