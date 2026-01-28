# Architecture

## Game Representation

Reused from `../damas`:

```cpp
using Bb = std::uint32_t;  // 32-bit bitboard for 32 playable squares

struct Board {
  Bb white;           // White piece positions
  Bb black;           // Black piece positions
  Bb kings;           // Queen positions (both colors)
  unsigned n_reversible;  // Moves without capture (for draw detection)
};

struct Move {
  Bb from_xor_to;     // XOR of source and destination
  Bb captures;        // Bitboard of captured pieces
};
```

**Key convention**: Board is flipped after each move so white is always to move.

## Search Engine

### Alpha-Beta with Iterative Deepening

```cpp
class Searcher {
public:
  SearchResult search(const Board& board, int max_depth, int time_limit_ms);

private:
  int alpha_beta(const Board& board, int depth, int alpha, int beta,
                 Accumulator& acc, bool is_pv);
  int quiescence(const Board& board, int alpha, int beta, Accumulator& acc);

  TranspositionTable tt_;
  const NNUEWeights* weights_;
  CompressedTablebaseManager* tb_;
};
```

### Transposition Table

```cpp
struct TTEntry {
  uint64_t key;       // Position hash (from Board::hash())
  int16_t score;      // Evaluation
  int8_t depth;       // Search depth
  uint8_t flag;       // EXACT, LOWER_BOUND, UPPER_BOUND
  Move best_move;     // Best move for move ordering
};
```

## NNUE Evaluation

### Architecture

```
Board Position
      │
      ▼
┌─────────────────────────────────────┐
│  Embedding Table [4][32][64]        │  16,384 bytes
│  4 piece types × 32 squares × 64dim │
└─────────────────────────────────────┘
      │
      ▼  Sum embeddings of active pieces
┌─────────────────────────────────────┐
│  Accumulator [64]                   │  Incrementally updated
└─────────────────────────────────────┘
      │
      ▼  ReLU
┌─────────────────────────────────────┐
│  FC Layer: 64 → 32                  │  4,160 bytes (weights + bias)
└─────────────────────────────────────┘
      │
      ▼  ReLU
┌─────────────────────────────────────┐
│  FC Layer: 32 → 3                   │  198 bytes (weights + bias)
└─────────────────────────────────────┘
      │
      ▼  Softmax (or approximation)
┌─────────────────────────────────────┐
│  Output: [P(loss), P(draw), P(win)] │
└─────────────────────────────────────┘
      │
      ▼
  Eval = P(win) - P(loss)
```

### Piece Type Encoding

```cpp
enum PieceType {
  WHITE_PAWN = 0,   // white & ~kings
  WHITE_QUEEN = 1,  // white & kings
  BLACK_PAWN = 2,   // black & ~kings
  BLACK_QUEEN = 3   // black & kings
};
```

### Incremental Accumulator

```cpp
struct Accumulator {
  alignas(32) int16_t values[64];

  void reset() { memset(values, 0, sizeof(values)); }

  void add_piece(PieceType type, int square, const NNUEWeights& w) {
    // AVX2: add embedding[type][square] to values
    for (int i = 0; i < 64; i += 16) {
      __m256i acc = _mm256_load_si256((__m256i*)(values + i));
      __m256i emb = _mm256_load_si256((__m256i*)(w.embeddings[type][square] + i));
      _mm256_store_si256((__m256i*)(values + i), _mm256_add_epi16(acc, emb));
    }
  }

  void remove_piece(PieceType type, int square, const NNUEWeights& w) {
    // Same but subtract
  }
};
```

### Integer Arithmetic

All computations use int16_t with fixed-point scaling:
- Embeddings scaled by SCALE=64
- After FC1: shift right to avoid overflow
- After FC2: shift right, then compute eval

```cpp
int evaluate(const Accumulator& acc, const NNUEWeights& w) {
  alignas(32) int32_t hidden[32];

  // FC1: ReLU(accumulator × fc1_weights + fc1_bias)
  for (int i = 0; i < 32; i++) {
    int32_t sum = w.fc1_bias[i];
    for (int j = 0; j < 64; j++) {
      sum += acc.values[j] * w.fc1_weights[j][i];
    }
    hidden[i] = std::max(0, sum >> SHIFT_FC1);
  }

  // FC2: hidden × fc2_weights + fc2_bias
  int32_t logits[3];
  for (int i = 0; i < 3; i++) {
    int32_t sum = w.fc2_bias[i];
    for (int j = 0; j < 32; j++) {
      sum += hidden[j] * w.fc2_weights[j][i];
    }
    logits[i] = sum >> SHIFT_FC2;
  }

  // Approximate: eval ≈ logits[WIN] - logits[LOSS]
  // Or use fixed-point softmax for more accuracy
  return logits[2] - logits[0];
}
```

## Self-Play Data Generation

### Game Flow

```
1. Initialize board to starting position
2. Play 10 uniform random legal moves (opening randomization)
3. While game not over:
   a. Search for best move
   b. If position is quiet (no captures):
      - Record (board, pre_tactical_flag, side_to_move)
   c. Make move
4. Label all recorded positions with game outcome
5. Write to HDF5
```

### Quiet Position Detection

```cpp
bool is_quiet(const Board& b) {
  return !has_captures(b);
}
```

### Pre-Tactical Flag

A position is "pre-tactical" if the move played leads to a position where
the opponent has captures available. These positions may be at the start
of a tactical sequence, so we might want to exclude them from training.

```cpp
bool is_pre_tactical(const Board& before, const Move& move) {
  Board after = makeMove(before, move);
  return has_captures(after);  // After flip, white (originally black) has captures
}
```

### HDF5 Output Format

```
training_data.h5
├── boards         # uint32[N, 4] - white, black, kings, n_reversible
├── pre_tactical   # uint8[N]     - 0 or 1
└── outcomes       # int8[N]      - -1 (loss), 0 (draw), +1 (win)
```

## Training Pipeline

### PyTorch Model

```python
class DamasNNUE(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.embeddings = nn.Embedding(4 * 32, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, piece_indices):
        # piece_indices: [batch, num_pieces] - indices into embedding table
        emb = self.embeddings(piece_indices)  # [batch, num_pieces, 64]
        acc = emb.sum(dim=1)                   # [batch, 64]
        h = torch.relu(self.fc1(acc))          # [batch, 32]
        logits = self.fc2(h)                   # [batch, 3]
        return logits
```

### Training Loop

```python
def train(model, dataloader, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for boards, outcomes in dataloader:
            features = board_to_features(boards)  # Convert to piece indices
            logits = model(features)

            # outcomes: -1, 0, +1 → labels: 0, 1, 2
            labels = outcomes + 1
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Weight Export

Quantize float32 weights to int16:

```python
SCALE = 64

def export(model, path):
    with open(path, 'wb') as f:
        # Embeddings: [128, 64] → reshape to [4, 32, 64]
        emb = (model.embeddings.weight.data * SCALE).round().clamp(-32768, 32767)
        f.write(emb.numpy().astype(np.int16).tobytes())

        # FC1 weights and bias
        fc1_w = (model.fc1.weight.data.T * SCALE).round().clamp(-32768, 32767)
        fc1_b = (model.fc1.bias.data * SCALE * SCALE).round().clamp(-32768, 32767)
        f.write(fc1_w.numpy().astype(np.int16).tobytes())
        f.write(fc1_b.numpy().astype(np.int16).tobytes())

        # FC2 weights and bias
        fc2_w = (model.fc2.weight.data.T * SCALE).round().clamp(-32768, 32767)
        fc2_b = (model.fc2.bias.data * SCALE * SCALE).round().clamp(-32768, 32767)
        f.write(fc2_w.numpy().astype(np.int16).tobytes())
        f.write(fc2_b.numpy().astype(np.int16).tobytes())
```

## Tournament System

### SPRT (Sequential Probability Ratio Test)

Test whether new model is significantly stronger:
- H0: Elo difference ≤ 0 (no improvement)
- H1: Elo difference ≥ 10 (meaningful improvement)
- α = β = 0.05 (5% false positive/negative rate)

```cpp
class SPRT {
public:
  SPRT(double elo0 = 0.0, double elo1 = 10.0,
       double alpha = 0.05, double beta = 0.05);

  void add_result(int result);  // +1 win, 0 draw, -1 loss

  enum Status { CONTINUE, ACCEPT_H0, ACCEPT_H1 };
  Status status() const;
  double llr() const;  // Log-likelihood ratio

private:
  double lower_bound_;  // ln(beta / (1 - alpha))
  double upper_bound_;  // ln((1 - beta) / alpha)
  int wins_, draws_, losses_;
};
```

### Tournament Flow

```
1. Initialize SPRT
2. For game = 1 to max_games:
   a. Play game (new model vs baseline, alternating colors)
   b. Record result
   c. Update SPRT
   d. If SPRT decides: return result
3. Return inconclusive (or use current LLR)
```

## Tablebase Integration

Use `CompressedTablebaseManager` from `../damas`:

```cpp
class TBProbe {
public:
  TBProbe(const std::string& tb_directory);

  // Returns evaluation in centipawns or UNKNOWN
  // WIN = +10000, LOSS = -10000, DRAW = 0
  int probe(const Board& board);

  bool is_available(const Board& board);

private:
  CompressedTablebaseManager manager_;
};
```

Integration in search:
```cpp
int alpha_beta(...) {
  // At leaf or when piece count is low enough
  if (tb_probe_.is_available(board)) {
    return tb_probe_.probe(board);
  }
  // ... normal evaluation
}
```
