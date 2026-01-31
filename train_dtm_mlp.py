#!/usr/bin/env python3
"""Train an MLP to predict DTM classes from board positions.

Uses the DTM sampler to stream positions from tablebases.
No overfitting possible since we sample from 21+ billion positions.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import dtm_sampler


# DTM class to WDL mapping:
# Classes 0-6 (WIN_*) -> WDL class 0 (WIN)
# Class 7 (DRAW) -> WDL class 1 (DRAW)
# Classes 8-14 (LOSS_*) -> WDL class 2 (LOSS)
DTM_TO_WDL = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2])


class CombinedLoss(nn.Module):
    """Combined WDL + DTM classification loss.

    Aggregates 15-class logits into 3-class WDL probabilities and computes
    both coarse (WDL) and fine (DTM) classification losses.
    """

    def __init__(self, wdl_weight=2.0, dtm_weight=1.0):
        super().__init__()
        self.wdl_weight = wdl_weight
        self.dtm_weight = dtm_weight

        # Aggregation matrix: WDL_probs = softmax(logits) @ agg_matrix
        # Shape: [15, 3] - sums probabilities for each WDL class
        agg = torch.zeros(15, 3)
        agg[0:7, 0] = 1.0    # WIN classes -> WDL WIN
        agg[7, 1] = 1.0      # DRAW -> WDL DRAW
        agg[8:15, 2] = 1.0   # LOSS classes -> WDL LOSS
        self.register_buffer('agg_matrix', agg)

    def forward(self, logits, dtm_labels):
        """
        Args:
            logits: [batch, 15] raw network output
            dtm_labels: [batch] DTM class labels (0-14)

        Returns:
            Combined loss scalar
        """
        # Fine-grained DTM loss
        dtm_loss = F.cross_entropy(logits, dtm_labels)

        # Aggregate to WDL probabilities
        probs = F.softmax(logits, dim=1)  # [batch, 15]
        wdl_probs = probs @ self.agg_matrix  # [batch, 3]

        # Convert DTM labels to WDL labels
        wdl_labels = DTM_TO_WDL.to(dtm_labels.device)[dtm_labels]

        # WDL loss (using log of aggregated probs)
        wdl_log_probs = torch.log(wdl_probs + 1e-8)
        wdl_loss = F.nll_loss(wdl_log_probs, wdl_labels)

        return self.wdl_weight * wdl_loss + self.dtm_weight * dtm_loss


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
    """Train for one epoch (epoch_size samples).

    Returns: (loss, dtm_accuracy, wdl_accuracy)
    """
    model.train()
    total_loss = 0
    dtm_correct = 0
    wdl_correct = 0
    total = 0

    n_batches = epoch_size // batch_size
    dtm_to_wdl = DTM_TO_WDL.to(device)

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

        # DTM accuracy (15-class)
        _, predicted = outputs.max(1)
        dtm_correct += predicted.eq(labels).sum().item()

        # WDL accuracy (3-class)
        pred_wdl = dtm_to_wdl[predicted]
        true_wdl = dtm_to_wdl[labels]
        wdl_correct += pred_wdl.eq(true_wdl).sum().item()

        total += len(labels)

    return total_loss / total, dtm_correct / total, wdl_correct / total


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
    parser.add_argument('--wdl-weight', type=float, default=2.0,
                        help='Weight for WDL loss component')
    parser.add_argument('--dtm-weight', type=float, default=1.0,
                        help='Weight for DTM loss component')
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

    criterion = CombinedLoss(wdl_weight=args.wdl_weight, dtm_weight=args.dtm_weight)
    criterion = criterion.to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"Loss weights: WDL={args.wdl_weight}, DTM={args.dtm_weight}")
    print(f"Epoch size: {args.epoch_size:,} samples")
    if args.epochs == 0:
        print("Training indefinitely (Ctrl+C to stop)")
    print(f"\n{'Epoch':>5} {'Loss':>10} {'DTM Acc':>10} {'WDL Acc':>10}")
    print("-" * 40)

    epoch = 0
    try:
        while args.epochs == 0 or epoch < args.epochs:
            epoch += 1
            loss, dtm_acc, wdl_acc = train_epoch(
                model, sampler, optimizer, criterion, args.device,
                epoch_size=args.epoch_size, batch_size=args.batch_size
            )

            print(f"{epoch:>5} {loss:>10.4f} {dtm_acc:>10.2%} {wdl_acc:>10.2%}")

            # Save checkpoint every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'hidden_sizes': hidden_sizes,
                'loss': loss,
                'dtm_accuracy': dtm_acc,
                'wdl_accuracy': wdl_acc,
            }, args.output)
    except KeyboardInterrupt:
        print(f"\nTraining interrupted at epoch {epoch}")

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
