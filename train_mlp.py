#!/usr/bin/env python3
"""Train an MLP to predict game outcomes from board positions.

Loads all training data to GPU VRAM for fast epoch iteration.
"""

import argparse
import glob
import struct
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


# Bit masks for extracting 32 bits, precomputed once on GPU
_bit_masks = None

def get_bit_masks(device):
    global _bit_masks
    if _bit_masks is None or _bit_masks.device != device:
        _bit_masks = (1 << torch.arange(32, device=device)).unsqueeze(0)  # (1, 32)
    return _bit_masks


def boards_to_features(boards, device):
    """Vectorized board-to-feature conversion on GPU.

    boards: (N, 4) int32 tensor [white, black, kings, n_reversible]
    Returns: (N, 128) float32 tensor
    """
    masks = get_bit_masks(device)  # (1, 32)

    white = boards[:, 0:1]  # (N, 1)
    black = boards[:, 1:2]
    kings = boards[:, 2:3]

    # Expand bits: (N, 1) & (1, 32) -> (N, 32) bool -> float
    white_bits = (white & masks).ne(0)
    black_bits = (black & masks).ne(0)
    kings_bits = (kings & masks).ne(0)

    white_men = (white_bits & ~kings_bits).float()
    white_kings = (white_bits & kings_bits).float()
    black_men = (black_bits & ~kings_bits).float()
    black_kings = (black_bits & kings_bits).float()

    return torch.cat([white_men, white_kings, black_men, black_kings], dim=1)


def normalize_scores(raw_scores):
    """Normalize engine search scores to [-1, +1].

    Mirrors score_to_normalized() from the C++ engine:
      Proven draws (|score| <= 10000) → 0
      Undecided (|score| in 10001..28000) → strip ±10000 offset → [-10000, +10000]
      Special (TB win/mate, |score| > 28000) → ±10000
    Then divide by 10000 to get [-1, +1].
    """
    SCORE_DRAW = 10000
    SCORE_SPECIAL = 28000
    scores = raw_scores.astype(np.float32)

    result = np.zeros_like(scores)

    # Special scores (TB win/mate) → ±10000
    result[scores > SCORE_SPECIAL] = SCORE_DRAW
    result[scores < -SCORE_SPECIAL] = -SCORE_DRAW

    # Undecided: strip ±10000 offset
    undecided_pos = (scores > SCORE_DRAW) & (scores <= SCORE_SPECIAL)
    undecided_neg = (scores < -SCORE_DRAW) & (scores >= -SCORE_SPECIAL)
    result[undecided_pos] = scores[undecided_pos] - SCORE_DRAW
    result[undecided_neg] = scores[undecided_neg] + SCORE_DRAW

    # Proven draws: already 0 from initialization

    return result / SCORE_DRAW  # Scale to [-1, +1]


def load_data_to_gpu(h5_files, device):
    """Load all training data from HDF5 files directly to GPU."""
    all_boards = []
    all_outcomes = []
    all_scores = []
    has_scores = False

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

            if 'scores' in f:
                scores = f['scores'][:]
                if 'pre_tactical' in f:
                    scores = scores[mask]
                all_scores.append(scores)
                has_scores = True
            else:
                all_scores.append(None)

    boards_np = np.concatenate(all_boards, axis=0)
    outcomes_np = np.concatenate(all_outcomes, axis=0)

    # Convert outcomes from {-1, 0, 1} to {0, 1, 2} for cross-entropy
    labels_np = (outcomes_np + 1).astype(np.int64)

    n = len(labels_np)
    print(f"Loaded {n} positions from {len(h5_files)} files")
    print(f"  Wins:   {np.sum(labels_np == 2):>8} ({100*np.mean(labels_np == 2):.1f}%)")
    print(f"  Draws:  {np.sum(labels_np == 1):>8} ({100*np.mean(labels_np == 1):.1f}%)")
    print(f"  Losses: {np.sum(labels_np == 0):>8} ({100*np.mean(labels_np == 0):.1f}%)")

    # Process scores if any files had them
    scores_gpu = None
    if has_scores:
        # For files without scores, fill with NaN (will be masked out in loss)
        score_parts = []
        offset = 0
        for i, s in enumerate(all_scores):
            file_n = len(all_boards[i])
            if s is not None:
                score_parts.append(normalize_scores(s))
            else:
                score_parts.append(np.full(file_n, np.nan, dtype=np.float32))
            offset += file_n
        scores_np = np.concatenate(score_parts, axis=0)
        valid_count = np.sum(~np.isnan(scores_np))
        print(f"  Scores: {valid_count} positions have search scores")
    else:
        scores_np = None

    print("Moving data to GPU...")
    boards_gpu = torch.from_numpy(boards_np.astype(np.int32)).to(device)
    labels_gpu = torch.from_numpy(labels_np).to(device)
    if scores_np is not None:
        scores_gpu = torch.from_numpy(scores_np).to(device)

    del boards_np, outcomes_np, labels_np, all_boards, all_outcomes, all_scores
    print(f"GPU memory: {torch.cuda.memory_allocated(device) / 1e6:.0f} MB")

    return boards_gpu, labels_gpu, scores_gpu


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


def score_loss_fn(outputs, target_scores):
    """MSE between model's expected value and normalized search score.

    outputs: (B, 3) raw logits [loss, draw, win]
    target_scores: (B,) normalized scores in [-1, +1]
    Returns: scalar loss (only over positions with valid scores)
    """
    valid = ~torch.isnan(target_scores)
    if not valid.any():
        return torch.tensor(0.0, device=outputs.device)

    probs = torch.softmax(outputs[valid], dim=1)
    expected_value = probs[:, 2] - probs[:, 0]  # P(win) - P(loss), in [-1, +1]
    return torch.nn.functional.mse_loss(expected_value, target_scores[valid])


def train_epoch(model, boards, labels, batch_size, optimizer, criterion, device,
                scores=None, score_weight=0.0):
    model.train()
    n = len(labels)
    perm = torch.randperm(n, device=device)
    total_loss = 0
    total_score_loss = 0
    correct = 0

    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        features = boards_to_features(boards[idx], device)
        batch_labels = labels[idx]

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, batch_labels)

        if scores is not None and score_weight > 0:
            sl = score_loss_fn(outputs, scores[idx])
            total_score_loss += sl.item() * len(batch_labels)
            loss = loss + score_weight * sl

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_labels)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_labels).sum().item()

    return total_loss / n, correct / n, total_score_loss / n


def evaluate(model, boards, labels, batch_size, criterion, device,
             scores=None, score_weight=0.0):
    model.eval()
    n = len(labels)
    total_loss = 0
    total_score_loss = 0
    correct = 0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            features = boards_to_features(boards[start:start + batch_size], device)
            batch_labels = labels[start:start + batch_size]

            outputs = model(features)
            loss = criterion(outputs, batch_labels)

            if scores is not None and score_weight > 0:
                sl = score_loss_fn(outputs, scores[start:start + batch_size])
                total_score_loss += sl.item() * len(batch_labels)
                loss = loss + score_weight * sl

            total_loss += loss.item() * len(batch_labels)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_labels).sum().item()

    return total_loss / n, correct / n, total_score_loss / n


def main():
    parser = argparse.ArgumentParser(description='Train MLP for checkers outcome prediction')
    parser.add_argument('--data', type=str, default='data/*.h5', help='Glob pattern for HDF5 files')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden', type=str, default='256,128,64', help='Hidden layer sizes')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split')
    parser.add_argument('--output', type=str, default='model.pt', help='Output model path')
    parser.add_argument('--score-weight', type=float, default=0.0,
                        help='Weight for search score distillation loss (0 = disabled)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Parse hidden layer sizes
    hidden_sizes = [int(x) for x in args.hidden.split(',')]

    # Load data
    h5_files = sorted(glob.glob(args.data))
    if not h5_files:
        print(f"No files found matching {args.data}")
        return

    device = torch.device(args.device)
    print(f"Loading from {len(h5_files)} files...")
    boards_gpu, labels_gpu, scores_gpu = load_data_to_gpu(h5_files, device)

    if args.score_weight > 0 and scores_gpu is None:
        print("Warning: --score-weight specified but no files contain scores, ignoring")
        args.score_weight = 0.0

    # Split into train/val
    n = len(labels_gpu)
    n_val = int(n * args.val_split)
    n_train = n - n_val

    # Deterministic shuffle for split
    gen = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=gen, device='cpu')
    train_idx = perm[:n_train].to(device)
    val_idx = perm[n_train:].to(device)

    train_boards = boards_gpu[train_idx]
    train_labels = labels_gpu[train_idx]
    val_boards = boards_gpu[val_idx]
    val_labels = labels_gpu[val_idx]

    train_scores = scores_gpu[train_idx] if scores_gpu is not None else None
    val_scores = scores_gpu[val_idx] if scores_gpu is not None else None

    # Free the unsplit data
    del boards_gpu, labels_gpu, scores_gpu, train_idx, val_idx
    torch.cuda.empty_cache()

    print(f"Train: {n_train}, Val: {n_val}")
    if args.score_weight > 0:
        print(f"Score distillation weight: {args.score_weight}")

    # Create model
    model = CheckersMLP(hidden_sizes=hidden_sizes)
    model = model.to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Hidden layers: {hidden_sizes}")
    print(f"Device: {device}")
    print(f"GPU memory: {torch.cuda.memory_allocated(device) / 1e6:.0f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_acc = 0
    sw = args.score_weight
    if sw > 0:
        print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} {'Score MSE':>10} {'Val Loss':>12} {'Val Acc':>10} {'VScore MSE':>10}")
        print("-" * 75)
    else:
        print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10}")
        print("-" * 55)

    for epoch in range(args.epochs):
        train_loss, train_acc, train_sl = train_epoch(
            model, train_boards, train_labels, args.batch_size, optimizer, criterion, device,
            scores=train_scores, score_weight=sw)
        val_loss, val_acc, val_sl = evaluate(
            model, val_boards, val_labels, args.batch_size, criterion, device,
            scores=val_scores, score_weight=sw)

        #scheduler.step(val_loss)

        if sw > 0:
            print(f"{epoch+1:>5} {train_loss:>12.4f} {train_acc:>10.2%} {train_sl:>10.4f} {val_loss:>12.4f} {val_acc:>10.2%} {val_sl:>10.4f}")
        else:
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
