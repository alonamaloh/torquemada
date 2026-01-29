#!/usr/bin/env python3
"""Export PyTorch model to simple binary format for C++ inference."""

import argparse
import struct
import torch
import numpy as np


def export_model(input_path, output_path):
    """Export model weights to binary format.

    Format:
      - uint32: number of layers (n)
      - For each layer:
        - uint32: input_size
        - uint32: output_size
        - float32[output_size * input_size]: weights (row-major)
        - float32[output_size]: biases
    """
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=True)
    state_dict = checkpoint['model_state_dict']
    hidden_sizes = checkpoint['hidden_sizes']

    # Collect layer parameters
    layers = []
    i = 0
    while f'net.{i}.weight' in state_dict:
        weight = state_dict[f'net.{i}.weight'].numpy()  # [out, in]
        bias = state_dict[f'net.{i}.bias'].numpy()      # [out]
        layers.append((weight, bias))
        i += 2  # Skip ReLU layers in Sequential

    print(f"Exporting {len(layers)} layers:")
    for idx, (w, b) in enumerate(layers):
        print(f"  Layer {idx}: {w.shape[1]} -> {w.shape[0]}")

    with open(output_path, 'wb') as f:
        # Number of layers
        f.write(struct.pack('<I', len(layers)))

        for weight, bias in layers:
            out_size, in_size = weight.shape
            f.write(struct.pack('<II', in_size, out_size))

            # Weights in row-major order (already is in PyTorch)
            f.write(weight.astype(np.float32).tobytes())
            f.write(bias.astype(np.float32).tobytes())

    print(f"Exported to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='PyTorch model file (.pt)')
    parser.add_argument('output', help='Output binary file (.bin)')
    args = parser.parse_args()

    export_model(args.input, args.output)
