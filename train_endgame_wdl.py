#!/usr/bin/env python3
"""Train a WDL model for 6-7 piece endgames using DTM tablebases.

Uses the same architecture as the 8+ piece model (3-class WDL).
Streams positions from DTM tablebases with perfect labels.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dtm_sampler


class WDLMlp(nn.Module):
    """MLP for WDL prediction (same as 8+ piece model)."""

    def __init__(self, hidden_sizes=[256, 128, 64]):
        super().__init__()

        layers = []
        in_size = 128  # 4 planes * 32 squares

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, 3))  # 3 classes: loss, draw, win

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, sampler, optimizer, criterion, device,
                epoch_size=10_000_000, batch_size=4096):
    """Train for one epoch (epoch_size samples)."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    n_batches = epoch_size // batch_size

    for _ in range(n_batches):
        features, labels = sampler.sample_batch(batch_size)

        features = torch.from_numpy(features).to(device)
        labels = torch.from_numpy(labels).long().to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train WDL model for 6-7 piece endgames')
    parser.add_argument('--dtm-dir', type=str, default='../damas',
                        help='Directory containing DTM tablebase files')
    parser.add_argument('--min-pieces', type=int, default=6,
                        help='Minimum piece count to train on')
    parser.add_argument('--max-pieces', type=int, default=7,
                        help='Maximum piece count to train on')
    parser.add_argument('--epochs', type=int, default=0,
                        help='Number of epochs (0 = infinite)')
    parser.add_argument('--epoch-size', type=int, default=10_000_000,
                        help='Samples per epoch')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden', type=str, default='256,128,64',
                        help='Hidden layer sizes')
    parser.add_argument('--output', type=str, default='endgame_wdl.pt',
                        help='Output model path')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    # Load DTM tablebases (filtered by piece count)
    print(f"Loading DTM tablebases from {args.dtm_dir}...")
    print(f"Piece count filter: {args.min_pieces}-{args.max_pieces}")
    sampler = dtm_sampler.DTMSampler()
    sampler.load(args.dtm_dir, args.min_pieces, args.max_pieces)
    print(f"Loaded {sampler.num_tables()} tables, {sampler.total_positions():,} positions")

    # Create model
    model = WDLMlp(hidden_sizes=hidden_sizes)
    model = model.to(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Device: {args.device}")
    print(f"Epoch size: {args.epoch_size:,} samples")
    if args.epochs == 0:
        print("Training indefinitely (Ctrl+C to stop)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{'Epoch':>5} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 30)

    epoch = 0
    try:
        while args.epochs == 0 or epoch < args.epochs:
            epoch += 1
            loss, acc = train_epoch(
                model, sampler, optimizer, criterion, args.device,
                epoch_size=args.epoch_size, batch_size=args.batch_size
            )

            print(f"{epoch:>5} {loss:>10.4f} {acc:>10.2%}")

            # Save checkpoint every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'hidden_sizes': hidden_sizes,
                'loss': loss,
                'accuracy': acc,
            }, args.output)
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")

    print(f"\nModel saved to {args.output}")


if __name__ == '__main__':
    main()
