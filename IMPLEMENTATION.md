# Implementation Plan

## Status Summary

This document tracks the implementation of the Torquemada Spanish Checkers engine. The project has evolved beyond the original training-focused plan to include a full web-based UI with WASM support.

### Completed Milestones

- ✅ **Core Engine** - Board representation, move generation, alpha-beta search with iterative deepening
- ✅ **NNUE Evaluation** - Neural network evaluation with incremental accumulator updates
- ✅ **Tablebase Integration** - DTM (Distance-to-Mate) tablebase probing for endgames up to 5 pieces
- ✅ **Self-Play Infrastructure** - Position generation with HDF5 output
- ✅ **Python Training** - PyTorch training pipeline with weight export
- ✅ **Web/WASM Build** - Complete browser-based UI with WebAssembly engine
- ✅ **GitHub Pages Hosting** - Live deployment at https://alonamaloh.github.io/torquemada/

### Remaining Work

- ❌ **Tournament System** - SPRT-based model comparison (partially implemented in match.cpp)
- ❌ **Automated Training Loop** - End-to-end training pipeline script

---

## Phase 1: Core Foundation ✅ COMPLETE

### Completed Tasks

1. **Board representation** - Bitboard-based board with white, black, and kings masks
2. **Move generation** - Full Spanish checkers rules including mandatory captures
3. **Alpha-beta search** - Iterative deepening with transposition table
4. **Move ordering** - Killer moves, history heuristic, MVV-LVA for captures

### Key Files

| File | Description |
|------|-------------|
| `core/board.hpp` | Bitboard-based board representation |
| `core/movegen.hpp` | Move generation with capture detection |
| `search/search.hpp` | Iterative deepening alpha-beta |
| `search/tt.hpp` | Transposition table |

---

## Phase 2: Self-Play Infrastructure ✅ COMPLETE

### Completed Tasks

1. **SelfPlayGenerator** - Generates training positions from self-play games
2. **DataWriter** - Writes positions to HDF5 format for Python training
3. **Random openings** - Randomized opening plies for position variety
4. **Pre-tactical filtering** - Marks positions near captures for training filtering

### Key Files

| File | Description |
|------|-------------|
| `selfplay/selfplay.hpp` | Self-play game generation |
| `selfplay/data_writer.hpp` | HDF5 dataset writer |
| `selfplay/selfplay_main.cpp` | Command-line self-play tool |

---

## Phase 3: NNUE Architecture ✅ COMPLETE

### Completed Tasks

1. **Embedding-based architecture** - Piece-type × square embeddings
2. **Incremental accumulator** - Efficient updates on make/unmake move
3. **WDL output** - Win/Draw/Loss probabilities
4. **Quantized weights** - int16 weights for fast inference
5. **Evaluation noise** - Small random noise for play variety

### Key Files

| File | Description |
|------|-------------|
| `nnue/nnue.hpp` | NNUE weight structures and forward pass |
| `nnue/accumulator.hpp` | Incremental accumulator |
| `training/model.py` | PyTorch model definition |

---

## Phase 4: Python Training ✅ COMPLETE

### Completed Tasks

1. **PyTorch model** - DamasNNUE with embedding + FC layers
2. **Data loader** - HDF5 dataset loading with pre-tactical filtering
3. **Training loop** - Cross-entropy loss training
4. **Weight export** - Quantized int16 weight export for C++
5. **DTM specialist model** - Separate model trained on tablebase positions

### Key Files

| File | Description |
|------|-------------|
| `training/model.py` | PyTorch NNUE model |
| `training/train.py` | Training script |
| `training/export_weights.py` | Weight quantization and export |

---

## Phase 5: Tournament System ⏳ PARTIAL

### Completed Tasks

1. **Match driver** - Basic engine vs engine match (`match.cpp`)
2. **Model comparison** - Can compare different weight files

### Remaining Tasks

- [ ] SPRT implementation for statistical significance
- [ ] Automated promotion of improved models
- [ ] ELO estimation

---

## Phase 6: Tablebase Integration ✅ COMPLETE

### Completed Tasks

1. **DTM tablebases** - 5-piece distance-to-mate tables
2. **Tablebase probing** - Integration into search tree
3. **Score scaling** - DTM values converted to centipawn scores
4. **Custom probe function** - Supports different tablebase backends (native/WASM)

### Key Files

| File | Description |
|------|-------------|
| `tablebase/dtm.hpp` | DTM tablebase interface |
| `tablebase/dtm_manager.hpp` | Multi-file tablebase management |

---

## Phase 7: Full Training Pipeline ⏳ NOT STARTED

### Remaining Tasks

- [ ] Master training loop script
- [ ] Sliding window training (last N generations)
- [ ] Automatic model promotion based on SPRT results
- [ ] Training metrics logging

---

## Phase 8: Web/WASM Build ✅ COMPLETE

*This phase was added beyond the original plan.*

### Completed Tasks

1. **Emscripten build** - Engine compiled to WebAssembly
2. **Web Worker** - Engine runs in background thread
3. **Board UI** - Canvas-based interactive board
4. **Move input** - Click-to-select, click-to-move interface
5. **Edit mode** - Custom position setup
6. **Engine controls** - Play as white/black/both, strength settings
7. **Move history** - Clickable move list with undo/redo
8. **Search info display** - Depth, score, nodes, PV during search
9. **Tablebase loading** - Download and cache 5-piece DTM tables
10. **Lazy tablebase loading** - On-demand 16KB chunk loading for fast startup
11. **GitHub Pages deployment** - Live hosting via gh-pages branch

### Key Files

| File | Description |
|------|-------------|
| `web/src/wasm/bindings.cpp` | Embind bindings for JavaScript |
| `web/dist/main.js` | Game controller and UI logic |
| `web/dist/board-ui.js` | Canvas board rendering |
| `web/dist/engine-worker.js` | Web Worker for engine communication |
| `web/dist/engine-api.js` | Main thread API for worker |
| `web/dist/index.html` | Web UI structure |
| `web/dist/style.css` | UI styling |

### Web Features

- **Piece animation** - Smooth moves with multi-capture delays
- **Board flipping** - Play from black's perspective
- **OPFS storage** - Persistent tablebase storage in browser
- **Progress callbacks** - Real-time search updates during thinking
- **Responsive design** - Works on various screen sizes

---

## Testing Checklist

- [x] Perft matches known values
- [x] Move generation handles all capture rules
- [x] Accumulator incremental == full recomputation
- [x] C++ eval matches Python eval (within quantization error)
- [x] HDF5 files have correct structure
- [x] Training loss decreases
- [x] Exported weights load correctly
- [ ] SPRT terminates correctly (random vs random ~50%)
- [x] Tablebase probes return correct values
- [ ] Full training pipeline runs end-to-end
- [x] WASM build works in browser
- [x] Tablebase lazy loading works correctly
- [x] Web UI is fully functional

---

## Git History Highlights

Recent development focused on the web/WASM build:

- Lazy chunk-based tablebase loading (eliminates 20-second startup delay)
- Edit mode for custom position setup
- New game dialog with play-as selection
- Multi-capture animation with delays
- Iterative deepening progress updates
- DTM tablebase integration in WASM search
- Full PV notation display with complete capture paths
