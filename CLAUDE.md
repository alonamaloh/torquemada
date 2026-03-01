# Notes for Claude

## Project Overview

Torquemada is a Spanish checkers (damas) engine with neural network evaluation, endgame tablebases, and a web interface. The engine improves through self-play reinforcement learning.

## Directory Layout

- **Main repo** (`/home/alvaro/claude/torquemada`): `master` branch, C++ source
- **gh-pages worktree** (`/home/alvaro/claude/torquemada-gh-pages`): deployed website (JS, HTML, CSS, WASM)
- **Tablebases** (`/home/alvaro/claude/damas`): DTM and CWDL tablebase files

### Source Modules

| Directory | Purpose |
|-----------|---------|
| `core/` | Board representation (32-bit bitboards), move generation, notation |
| `search/` | Alpha-beta search, iterative deepening, transposition table |
| `nn/` | MLP neural network inference (128→32→3, WDL output) |
| `tablebase/` | DTM probing (`tb_probe.hpp`), CWDL compression (`compression.hpp`), material indexing |
| `web/src/wasm/` | Emscripten bindings (`bindings.cpp`) |
| `web/dist/` | Built WASM output (`engine.js`, `engine.wasm`, `engine.worker.js`) |

### Key Conventions

- **Board always has white to move.** After each move the board is flipped 180° via `flip()`.
- **Square indexing:** 0-31 internal (bit positions), 1-32 human notation. Convert: `human_sq - 1 = bit_index`.
- **Score encoding:** NN eval in [-10000, +10000]. `SCORE_DRAW = 10000` (proven draw range). `SCORE_TB_WIN = 29000`. `SCORE_MATE = 30000`. `to_undecided(nn_score)` shifts by ±10000.
- **Tablebase coverage:** DTM files exist for ≤6 pieces (867 files, only 116 are 7-piece). CWDL (compressed WDL) covers all material configs up to 7 pieces.

## Build System

```bash
make all              # Build all native executables (into bin/)
make wasm             # Build WASM engine (into web/dist/)
make clean            # Clean native build
make wasm-clean       # Clean WASM build
```

Compiler: `g++ -std=c++20 -O3 -march=native -mbmi2 -mavx2`. WASM uses `em++` with `-msimd128 -pthread`.

### Executables

| Target | Purpose |
|--------|---------|
| `generate_training` | Self-play data generation (OpenMP, HDF5 output) |
| `print_games` | Print generated games as human-readable text |
| `match` | Engine vs engine tournament |
| `play` | Interactive CLI play |
| `genbook` | PUCT-based opening book generator |
| `viewbook` / `condensebook` | View or compress opening books |
| `test_search` / `perft` / `test_nn` | Testing tools |

### Building individual targets

```bash
make bin/generate_training    # Needs HDF5 and OpenMP
make bin/match                # Needs OpenMP
make bin/play
```

`print_games` and `test_draw_score` are not in the Makefile; build manually:
```bash
g++ -std=c++20 -O3 -march=native -Wall -Wextra -mbmi2 -mavx2 \
    -c print_games.cpp -o obj/print_games.o
g++ -O3 -o bin/print_games obj/print_games.o \
    obj/core/board.o obj/core/movegen.o obj/core/notation.o \
    obj/search/search.o obj/search/tt.o \
    obj/tablebase/tablebase.o obj/tablebase/compression.o obj/nn/mlp.o
```

## Training Pipeline

### Data Generation

```bash
bin/generate_training \
    --model ../torquemada-gh-pages/models/model_006.bin \
    --nodes 10000 --random-plies 10 --threads 16 \
    --positions 9000000 --output training_data_3.h5 \
    --tb-path /home/alvaro/claude/damas
```

- Games are played with random openings, then search with NN eval
- Adjudicated at ≤7 pieces using compressed WDL tablebases
- Only quiet positions recorded (no captures available on either side)
- HDF5 written incrementally; Ctrl+C saves collected data gracefully
- HDF5 format: `boards` (uint32[N,4]: white, black, kings, n_reversible), `outcomes` (int8[N]: -1/0/+1)

### Model Training

```bash
python3 train_mlp.py training_data_3.h5 --output model_007.bin
```

### Evaluation

```bash
bin/match model_007.bin model_006.bin --pairs 100 --nodes 10000 --threads 16
```

## Deployment

The web interface is at https://alonamaloh.github.io/torquemada/, deployed from the `gh-pages` branch.

### After changing C++ engine code

```bash
# 1. Build WASM
make wasm

# 2. Copy to gh-pages worktree and deploy
cp web/dist/engine.js web/dist/engine.wasm web/dist/engine.worker.js ../torquemada-gh-pages/
cd ../torquemada-gh-pages
git add engine.js engine.wasm engine.worker.js
git commit -m "Update engine: <description>"
git push origin gh-pages
cd ../torquemada
```

### After changing JavaScript UI

Edit files directly in `../torquemada-gh-pages/`, commit and push to `gh-pages`.

### Cache Busting

Automatic: `index.html` loads `main.js?v=${Date.now()}`, and each module propagates the version parameter via `import.meta.url`. Emscripten-generated files (`engine.wasm`, `engine.worker.js`) use `locateFile` in `engine-worker.js`. No manual version bumping needed.

### Web Architecture

| File | Role |
|------|------|
| `main.js` | Entry point, event wiring, UI updates |
| `engine-api.js` | Main-thread API for communicating with the engine worker |
| `engine-worker.js` | Web Worker: loads WASM, dispatches search |
| `search-manager.js` | Search lifecycle (thinking, pondering, time bank, PV) |
| `game-state.js` | Game history, undo/redo, position tracking |
| `board-ui.js` | Canvas board rendering and piece animation |
| `move-input.js` | Click-to-move interaction |
| `turn-controller.js` | Game flow: human move → engine search → result |
| `tablebase-loader.js` | OPFS-based persistent tablebase caching |
| `game-storage.js` | LocalStorage for saved games |

### Worktree Management

```bash
git worktree list                              # List worktrees
git worktree add ../torquemada-gh-pages gh-pages  # Recreate if needed
```
