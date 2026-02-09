#!/usr/bin/env python3
"""Train an MLP to predict game outcomes from board positions."""

import argparse
import glob
import struct
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def export_to_bin(model, output_path):
    """Export model weights to binary format for C++ inference."""
    state_dict = model.state_dict()

    # Collect layer parameters
    layers = []
    i = 0
    while f'net.{i}.weight' in state_dict:
        weight = state_dict[f'net.{i}.weight'].cpu().numpy()
        bias = state_dict[f'net.{i}.bias'].cpu().numpy()
        layers.append((weight, bias))
        i += 2  # Skip ReLU layers

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<I', len(layers)))
        for weight, bias in layers:
            out_size, in_size = weight.shape
            f.write(struct.pack('<II', in_size, out_size))
            f.write(weight.astype(np.float32).tobytes())
            f.write(bias.astype(np.float32).tobytes())


class CheckersDataset(Dataset):
    """Dataset for checkers positions with outcome labels."""

    def __init__(self, h5_files):
        """Load positions from HDF5 files."""
        all_boards = []
        all_outcomes = []

        for path in h5_files:
            with h5py.File(path, 'r') as f:
                boards = f['boards'][:]
                outcomes = f['outcomes'][:]

                # Filter out pre-tactical positions from old-format files
                if 'pre_tactical' in f:
                    mask = f['pre_tactical'][:] == 0
                    boards = boards[mask]
                    outcomes = outcomes[mask]

                all_boards.append(boards)
                all_outcomes.append(outcomes)

        self.boards = np.concatenate(all_boards, axis=0)
        self.outcomes = np.concatenate(all_outcomes, axis=0)

        # Convert outcomes from {-1, 0, 1} to {0, 1, 2} for cross-entropy
        # 0 = loss, 1 = draw, 2 = win
        self.labels = (self.outcomes + 1).astype(np.int64)

        print(f"Loaded {len(self.boards)} positions from {len(h5_files)} files")
        print(f"  Wins:   {np.sum(self.labels == 2):>8} ({100*np.mean(self.labels == 2):.1f}%)")
        print(f"  Draws:  {np.sum(self.labels == 1):>8} ({100*np.mean(self.labels == 1):.1f}%)")
        print(f"  Losses: {np.sum(self.labels == 0):>8} ({100*np.mean(self.labels == 0):.1f}%)")

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        features = self._board_to_features(board)
        label = self.labels[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label)

    def _board_to_features(self, board):
        """Convert board representation to feature vector.

        Board format: [white, black, kings, n_reversible]
        Each of white/black/kings is a 32-bit mask for the 32 playable squares.

        Output: 128 features
          - 32 for white men (white & ~kings)
          - 32 for white kings (white & kings)
          - 32 for black men (black & ~kings)
          - 32 for black kings (black & kings)
        """
        white = int(board[0])
        black = int(board[1])
        kings = int(board[2])

        white_men = white & ~kings
        white_kings = white & kings
        black_men = black & ~kings
        black_kings = black & kings

        features = np.zeros(128, dtype=np.float32)
        for i in range(32):
            mask = 1 << i
            features[i] = 1.0 if (white_men & mask) else 0.0
            features[32 + i] = 1.0 if (white_kings & mask) else 0.0
            features[64 + i] = 1.0 if (black_men & mask) else 0.0
            features[96 + i] = 1.0 if (black_kings & mask) else 0.0

        return features


class CheckersMLP(nn.Module):
    """Simple MLP for outcome prediction."""

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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

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


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += len(labels)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train MLP for checkers outcome prediction')
    parser.add_argument('--data', type=str, default='data/*.h5', help='Glob pattern for HDF5 files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=str, default='256,128,64', help='Hidden layer sizes')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--output', type=str, default='model.pt', help='Output model path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    # Load data
    h5_files = sorted(glob.glob(args.data))
    if not h5_files:
        print(f"No files found matching {args.data}")
        return

    print(f"Loading from {len(h5_files)} files...")
    dataset = CheckersDataset(h5_files)

    # Split into train/val
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Create model
    model = CheckersMLP(hidden_sizes=hidden_sizes)
    model = model.to(args.device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Device: {args.device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10}")
    print("-" * 55)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)

        scheduler.step(val_loss)

        print(f"{epoch+1:>5} {train_loss:>12.4f} {train_acc:>10.2%} {val_loss:>12.4f} {val_acc:>10.2%}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'hidden_sizes': hidden_sizes,
                'val_acc': val_acc,
            }, args.output)
            bin_path = args.output.replace('.pt', '.bin')
            export_to_bin(model, bin_path)

        # Save checkpoint every epoch (both .pt and .bin for C++ matches)
        checkpoint_path = args.output.replace('.pt', f'_epoch_{epoch+1}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'hidden_sizes': hidden_sizes,
            'epoch': epoch + 1,
            'val_acc': val_acc,
        }, checkpoint_path)
        bin_path = checkpoint_path.replace('.pt', '.bin')
        export_to_bin(model, bin_path)
        print(f"  -> Saved checkpoint: {checkpoint_path} and {bin_path}")

    print(f"\nBest validation accuracy: {best_val_acc:.2%}")
    print(f"Model saved to {args.output}")


if __name__ == '__main__':
    main()
