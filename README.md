# Torquemada: Spanish Checkers Self-Play Engine

A self-play reinforcement learning engine for Spanish checkers (damas) that iteratively improves through neural network training.

## Overview

This project builds a strong checkers engine by:
1. Starting with a random evaluation function
2. Generating self-play games with random openings for variety
3. Training a neural network to predict game outcomes
4. Replacing the evaluation function with the trained network
5. Repeating until no improvement is detected (via SPRT)

## Key Features

- **Alpha-Beta Search** with iterative deepening
- **NNUE-Style Evaluation**: Efficient integer arithmetic with SIMD (AVX2)
- **Incremental Updates**: Accumulator maintained during search tree traversal
- **Endgame Tablebases**: Integration with compressed WDL tablebases from `../damas`
- **PyTorch Training**: GPU-accelerated (CUDA) neural network training

## Directory Structure

```
torquemada/
├── core/               # Game logic (from damas)
├── nnue/               # Neural network evaluation
├── search/             # Alpha-beta search engine
├── selfplay/           # Self-play data generation
├── tablebase/          # Endgame tablebase integration
├── tournament/         # Model evaluation (SPRT)
├── training/           # Python training code
├── models/             # Saved model weights
├── data/               # Training data (HDF5)
└── Makefile
```

## Dependencies

**C++:**
- GCC with C++20 support
- x86-64 with BMI2 and AVX2
- HDF5 C++ library
- OpenMP (optional, for parallel self-play)

**Python:**
- PyTorch with CUDA
- h5py
- numpy

## Quick Start

```bash
# Build
make all

# Generate self-play data (random eval initially)
./bin/selfplay --positions 1000000 --output data/gen_001.h5

# Train neural network
python training/train.py --data data/gen_001.h5 --output models/gen_001/weights.bin

# Evaluate improvement
./bin/tournament --new models/gen_001/ --baseline random
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical architecture details
- [IMPLEMENTATION.md](IMPLEMENTATION.md) - Step-by-step implementation plan

## Related

- `../damas` - Endgame tablebase generator for Spanish checkers
