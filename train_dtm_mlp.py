#!/usr/bin/env python3
"""Train an MLP to predict DTM classes from board positions.

Uses the DTM sampler to stream positions from tablebases.
No overfitting possible since we sample from 21+ billion positions.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import dtm_sampler


class DTMMlp(nn.Module):
    """MLP for DTM class prediction."""

    def __init__(self, hidden_sizes=[256, 128, 64], num_classes=15):
        super().__init__()

        layers = []
        in_size = 128  # 4 planes * 32 squares

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size

        layers.append(nn.Linear(in_size, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_epoch(model, sampler, optimizer, criterion, device,
                epoch_size=1_000_000, batch_size=4096):
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


def compute_class_stats(model, sampler, device, n_samples=100_000, batch_size=4096):
    """Compute per-class accuracy."""
    model.eval()

    class_correct = np.zeros(dtm_sampler.NUM_DTM_CLASSES)
    class_total = np.zeros(dtm_sampler.NUM_DTM_CLASSES)

    n_batches = n_samples // batch_size

    with torch.no_grad():
        for _ in range(n_batches):
            features, labels = sampler.sample_batch(batch_size)

            features = torch.from_numpy(features).to(device)
            labels_t = torch.from_numpy(labels).long().to(device)

            outputs = model(features)
            _, predicted = outputs.max(1)

            for cls in range(dtm_sampler.NUM_DTM_CLASSES):
                mask = labels == cls
                class_total[cls] += mask.sum()
                class_correct[cls] += (predicted.cpu().numpy()[mask] == cls).sum()

    return class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description='Train MLP for DTM class prediction')
    parser.add_argument('--dtm-dir', type=str, default='../damas',
                        help='Directory containing DTM tablebase files')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--epoch-size', type=int, default=1_000_000,
                        help='Samples per epoch')
    parser.add_argument('--batch-size', type=int, default=4096,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden', type=str, default='256,128,64',
                        help='Hidden layer sizes')
    parser.add_argument('--output', type=str, default='dtm_model.pt',
                        help='Output model path')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    # Load DTM tablebases
    print(f"Loading DTM tablebases from {args.dtm_dir}...")
    sampler = dtm_sampler.DTMSampler()
    sampler.load(args.dtm_dir)
    print(f"Loaded {sampler.num_tables()} tables, {sampler.total_positions():,} positions")

    # Create model
    model = DTMMlp(hidden_sizes=hidden_sizes, num_classes=dtm_sampler.NUM_DTM_CLASSES)
    model = model.to(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Device: {args.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n{'Epoch':>5} {'Loss':>10} {'Accuracy':>10}")
    print("-" * 30)

    for epoch in range(args.epochs):
        loss, acc = train_epoch(
            model, sampler, optimizer, criterion, args.device,
            epoch_size=args.epoch_size, batch_size=args.batch_size
        )

        print(f"{epoch+1:>5} {loss:>10.4f} {acc:>10.2%}")

        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'hidden_sizes': hidden_sizes,
            'loss': loss,
            'accuracy': acc,
        }, args.output)

    # Final class-wise stats
    print("\nPer-class accuracy:")
    class_correct, class_total = compute_class_stats(
        model, sampler, args.device, n_samples=1_000_000
    )

    for cls in range(dtm_sampler.NUM_DTM_CLASSES):
        name = dtm_sampler.DTM_CLASS_NAMES[cls]
        if class_total[cls] > 0:
            acc = class_correct[cls] / class_total[cls]
            pct = 100 * class_total[cls] / class_total.sum()
            print(f"  {name:>12}: {acc:6.2%} ({pct:5.1f}% of samples)")

    print(f"\nModel saved to {args.output}")


if __name__ == '__main__':
    main()
