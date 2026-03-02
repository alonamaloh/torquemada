#!/usr/bin/env python3
"""Filter training data to exclude positions with 7 or fewer pieces."""

import argparse
import numpy as np
import h5py


def popcount(arr):
    """Vectorized popcount for uint32 arrays."""
    x = arr.astype(np.uint64)
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return ((x * 0x0101010101010101) >> 56).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Filter training data by piece count")
    parser.add_argument("inputs", nargs="+", help="Input .h5 files")
    parser.add_argument("-o", "--output", required=True, help="Output .h5 file")
    parser.add_argument("--min-pieces", type=int, default=8,
                        help="Minimum number of pieces to keep (default: 8)")
    args = parser.parse_args()

    all_boards = []
    all_outcomes = []
    all_scores = []
    has_scores = False

    for path in args.inputs:
        with h5py.File(path, "r") as f:
            boards = f["boards"][:]
            outcomes = f["outcomes"][:]
            white, black = boards[:, 0], boards[:, 1]
            n_pieces = popcount(white) + popcount(black)
            mask = n_pieces >= args.min_pieces
            kept = mask.sum()
            print(f"{path}: {len(boards)} positions, {kept} with >= {args.min_pieces} pieces")
            all_boards.append(boards[mask])
            all_outcomes.append(outcomes[mask])
            if "scores" in f:
                all_scores.append(f["scores"][:][mask])
                has_scores = True

    boards = np.concatenate(all_boards)
    outcomes = np.concatenate(all_outcomes)
    print(f"Total: {len(boards)} positions")

    with h5py.File(args.output, "w") as f:
        f.create_dataset("boards", data=boards)
        f.create_dataset("outcomes", data=outcomes)
        if has_scores:
            scores = np.concatenate(all_scores)
            f.create_dataset("scores", data=scores)
    print(f"Written to {args.output}")


if __name__ == "__main__":
    main()
