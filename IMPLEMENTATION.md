# Implementation Plan

## Phase 1: Core Foundation

### Tasks

1. **Copy game logic from damas**
   ```bash
   cp ../damas/board.h ../damas/board.cpp core/
   cp ../damas/movegen.h ../damas/movegen.cpp core/
   ```

2. **Create Makefile**
   ```makefile
   CXX = g++
   CXXFLAGS = -std=c++20 -O3 -march=native -Wall -Wextra -mbmi2 -mavx2
   LDFLAGS = -lhdf5_cpp -lhdf5 -lpthread

   all: bin/selfplay bin/tournament bin/test
   ```

3. **Implement random evaluation**
   ```cpp
   // search/eval.cpp
   int random_eval(std::mt19937& rng) {
     std::uniform_int_distribution<int> dist(-10000, 10000);
     return dist(rng);
   }
   ```

4. **Implement basic alpha-beta**
   ```cpp
   // search/search.cpp
   int alpha_beta(const Board& board, int depth, int alpha, int beta) {
     if (depth == 0) return evaluate(board);

     std::vector<Move> moves;
     generateMoves(board, moves);

     if (moves.empty()) return -MATE_SCORE;  // Loss

     for (const Move& move : moves) {
       Board child = flip(makeMove(board, move));
       int score = -alpha_beta(child, depth - 1, -beta, -alpha);
       if (score >= beta) return beta;
       if (score > alpha) alpha = score;
     }
     return alpha;
   }
   ```

### Verification

```bash
# Perft test (should match known values)
./bin/test --perft

# Known values for Spanish checkers:
# perft(1) = 9
# perft(2) = 81
# perft(3) = 658
# ...
```

---

## Phase 2: Self-Play Infrastructure

### Tasks

1. **Implement SelfPlayGenerator**
   ```cpp
   // selfplay/selfplay.h
   struct TrainingPosition {
     Board board;
     bool pre_tactical;
     int8_t outcome;  // Set after game ends
   };

   class SelfPlayGenerator {
   public:
     std::vector<TrainingPosition> play_game(int random_opening_plies = 10);
   };
   ```

2. **Implement DataWriter (HDF5)**
   ```cpp
   // selfplay/data_writer.cpp
   void DataWriter::write_positions(const std::vector<TrainingPosition>& positions) {
     // Extend datasets
     // Write board data as uint32[N, 4]
     // Write pre_tactical as uint8[N]
     // Write outcomes as int8[N]
   }
   ```

3. **Create selfplay_main.cpp**
   ```cpp
   int main(int argc, char** argv) {
     int target_positions = parse_arg("--positions", 1000000);
     std::string output = parse_arg("--output", "data.h5");

     SelfPlayGenerator gen;
     DataWriter writer(output);

     while (total_positions < target_positions) {
       auto positions = gen.play_game(10);
       writer.write_positions(positions);
       total_positions += positions.size();
     }
   }
   ```

### Verification

```bash
# Generate test dataset
./bin/selfplay --positions 10000 --output test.h5

# Validate with Python
python -c "
import h5py
with h5py.File('test.h5', 'r') as f:
    print('boards:', f['boards'].shape)
    print('outcomes:', f['outcomes'].shape)
    print('unique outcomes:', set(f['outcomes'][:]))
"
```

---

## Phase 3: NNUE Architecture (C++)

### Tasks

1. **Define weight structures**
   ```cpp
   // nnue/nnue.h
   constexpr int EMBEDDING_DIM = 64;

   struct NNUEWeights {
     int16_t embeddings[4][32][EMBEDDING_DIM];
     int16_t fc1_weights[EMBEDDING_DIM][32];
     int16_t fc1_bias[32];
     int16_t fc2_weights[32][3];
     int16_t fc2_bias[3];
   };

   bool load_weights(NNUEWeights& w, const std::string& path);
   ```

2. **Implement Accumulator**
   ```cpp
   // nnue/accumulator.h
   struct Accumulator {
     alignas(32) int16_t values[EMBEDDING_DIM];

     void init_from_board(const Board& b, const NNUEWeights& w);
     void add_piece(int type, int square, const NNUEWeights& w);
     void remove_piece(int type, int square, const NNUEWeights& w);
   };
   ```

3. **Implement forward pass**
   ```cpp
   // nnue/nnue.cpp
   int evaluate(const Accumulator& acc, const NNUEWeights& w);
   ```

4. **Add SIMD helpers**
   ```cpp
   // nnue/simd.h
   void vec_add_epi16(int16_t* dst, const int16_t* src, int n);
   void vec_sub_epi16(int16_t* dst, const int16_t* src, int n);
   ```

5. **Integrate into search**
   - Pass Accumulator through search tree
   - Update incrementally on make/unmake

### Verification

```bash
# Test accumulator consistency
./bin/test --nnue-accumulator

# Test against Python reference
./bin/test --nnue-eval test_weights.bin
```

---

## Phase 4: Python Training

### Tasks

1. **Create model.py**
   ```python
   # training/model.py
   class DamasNNUE(nn.Module):
       def __init__(self, embedding_dim=64):
           super().__init__()
           self.embeddings = nn.Embedding(4 * 32, embedding_dim)
           self.fc1 = nn.Linear(embedding_dim, 32)
           self.fc2 = nn.Linear(32, 3)

       def forward(self, piece_indices):
           emb = self.embeddings(piece_indices).sum(dim=1)
           h = torch.relu(self.fc1(emb))
           return self.fc2(h)
   ```

2. **Create data_loader.py**
   ```python
   # training/data_loader.py
   class DamasDataset(Dataset):
       def __init__(self, h5_path, exclude_pretactical=False):
           with h5py.File(h5_path, 'r') as f:
               self.boards = f['boards'][:]
               self.outcomes = f['outcomes'][:]
               self.pretactical = f['pre_tactical'][:]

           if exclude_pretactical:
               mask = self.pretactical == 0
               self.boards = self.boards[mask]
               self.outcomes = self.outcomes[mask]

       def __getitem__(self, idx):
           board = self.boards[idx]
           features = board_to_piece_indices(board)
           outcome = self.outcomes[idx] + 1  # -1,0,+1 → 0,1,2
           return features, outcome
   ```

3. **Create train.py**
   ```python
   # training/train.py
   def train(args):
       model = DamasNNUE().cuda()
       optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

       dataset = DamasDataset(args.data)
       loader = DataLoader(dataset, batch_size=4096, shuffle=True)

       for epoch in range(args.epochs):
           for features, labels in loader:
               logits = model(features.cuda())
               loss = F.cross_entropy(logits, labels.cuda())
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

       export_weights(model, args.output)
   ```

4. **Create export_weights.py**
   ```python
   # training/export_weights.py
   def export_weights(model, path, scale=64):
       with open(path, 'wb') as f:
           # Write embeddings, fc1, fc2 as int16
           ...
   ```

### Verification

```bash
# Train on test data
python training/train.py --data test.h5 --epochs 10 --output test.bin

# Load in C++ and verify
./bin/test --load-weights test.bin
```

---

## Phase 5: Tournament System

### Tasks

1. **Implement SPRT**
   ```cpp
   // tournament/sprt.h
   class SPRT {
   public:
     SPRT(double elo0 = 0, double elo1 = 10, double alpha = 0.05, double beta = 0.05);
     void add_result(int result);
     Status status() const;
   };
   ```

2. **Implement tournament driver**
   ```cpp
   // tournament/tournament.cpp
   SPRTResult run_tournament(
       const NNUEWeights& new_model,
       const NNUEWeights& baseline,
       int max_games = 10000
   );
   ```

3. **Create tournament_main.cpp**
   ```cpp
   int main(int argc, char** argv) {
     NNUEWeights new_model, baseline;
     load_weights(new_model, args.new_model);
     load_weights(baseline, args.baseline);

     auto result = run_tournament(new_model, baseline);
     // Print result and exit code
   }
   ```

### Verification

```bash
# Random vs random should be ~50%
./bin/tournament --new random --baseline random --games 1000

# Trained vs random should win significantly
./bin/tournament --new models/gen_001/weights.bin --baseline random
```

---

## Phase 6: Tablebase Integration

### Tasks

1. **Copy compression lookup API**
   ```bash
   cp ../damas/compression.h ../damas/compression.cpp tablebase/
   cp ../damas/tablebase.h ../damas/tablebase.cpp tablebase/
   ```

2. **Create tb_probe wrapper**
   ```cpp
   // tablebase/tb_probe.h
   class TBProbe {
   public:
     TBProbe(const std::string& directory);
     int probe(const Board& board);  // Returns centipawns or UNKNOWN
     bool available(const Board& board);
   private:
     CompressedTablebaseManager manager_;
   };
   ```

3. **Integrate into search**
   ```cpp
   // In alpha_beta, at leaves or low piece count:
   if (popcount(board.allPieces()) <= TB_PIECE_LIMIT) {
     if (tb_.available(board)) {
       return tb_.probe(board);
     }
   }
   ```

### Verification

```bash
# Probe known positions
./bin/test --tablebase-probe

# Verify search finds tablebase wins
./bin/test --search-with-tb
```

---

## Phase 7: Full Pipeline

### Master Training Script

```bash
#!/bin/bash
# train_loop.sh

GENERATION=1
BEST_MODEL="random"

while true; do
    echo "=== Generation $GENERATION ==="

    # Generate self-play data
    ./bin/selfplay \
        --model "$BEST_MODEL" \
        --positions 1000000 \
        --output "data/gen_$(printf '%03d' $GENERATION).h5"

    # Train (use sliding window of last 5 generations)
    python training/train.py \
        --data "data/gen_*.h5" \
        --window 5 \
        --epochs 100 \
        --output "models/gen_$(printf '%03d' $GENERATION)/weights.bin"

    # Evaluate
    RESULT=$(./bin/tournament \
        --new "models/gen_$(printf '%03d' $GENERATION)/weights.bin" \
        --baseline "$BEST_MODEL")

    if [[ "$RESULT" == "ACCEPT_H1" ]]; then
        echo "Improvement detected!"
        BEST_MODEL="models/gen_$(printf '%03d' $GENERATION)/weights.bin"
        GENERATION=$((GENERATION + 1))
    else
        echo "No improvement. Training complete."
        break
    fi
done
```

### Sliding Window Training

```python
# In train.py
def load_sliding_window(data_dir, window_size=5):
    files = sorted(glob.glob(f"{data_dir}/gen_*.h5"))[-window_size:]
    datasets = [DamasDataset(f) for f in files]
    return ConcatDataset(datasets)
```

---

## Files to Create

| File | Description |
|------|-------------|
| `core/board.h` | Copy from damas |
| `core/board.cpp` | Copy from damas |
| `core/movegen.h` | Copy from damas |
| `core/movegen.cpp` | Copy from damas |
| `nnue/nnue.h` | NNUE architecture |
| `nnue/nnue.cpp` | Forward pass |
| `nnue/accumulator.h` | Incremental accumulator |
| `nnue/simd.h` | AVX2 helpers |
| `search/search.h` | Search interface |
| `search/search.cpp` | Alpha-beta implementation |
| `search/tt.h` | Transposition table |
| `search/tt.cpp` | TT implementation |
| `search/eval.h` | Evaluation interface |
| `search/eval.cpp` | Random + NNUE eval |
| `selfplay/selfplay.h` | Self-play interface |
| `selfplay/selfplay.cpp` | Game generation |
| `selfplay/data_writer.h` | HDF5 writer |
| `selfplay/data_writer.cpp` | HDF5 implementation |
| `tablebase/tb_probe.h` | TB probe wrapper |
| `tablebase/tb_probe.cpp` | TB implementation |
| `tournament/sprt.h` | SPRT interface |
| `tournament/sprt.cpp` | SPRT implementation |
| `tournament/tournament.h` | Tournament interface |
| `tournament/tournament.cpp` | Tournament driver |
| `training/model.py` | PyTorch model |
| `training/train.py` | Training loop |
| `training/data_loader.py` | HDF5 data loading |
| `training/export_weights.py` | Weight export |
| `selfplay_main.cpp` | Self-play executable |
| `tournament_main.cpp` | Tournament executable |
| `test_main.cpp` | Test executable |
| `Makefile` | Build system |

---

## Testing Checklist

- [ ] Perft matches known values
- [ ] Move generation handles all capture rules
- [ ] Accumulator incremental == full recomputation
- [ ] C++ eval matches Python eval (within quantization error)
- [ ] HDF5 files have correct structure
- [ ] Training loss decreases
- [ ] Exported weights load correctly
- [ ] SPRT terminates correctly (random vs random ~50%)
- [ ] Tablebase probes return correct values
- [ ] Full pipeline runs end-to-end
