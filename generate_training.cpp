// Training data generator for neural network training.
//
// NOTE: This program only needs WDL (win/draw/loss) tablebases, NOT DTM
// (distance-to-mate) tablebases. DTM tables are for optimal play; here we
// only need WDL to determine game outcomes when positions reach tablebase
// territory. Do not add DTM preloading or usage here.

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"
#include <H5Cpp.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <vector>
#include <bit>

// Training position: board state + metadata
struct TrainingPosition {
  Board board;
  int ply;           // When this position occurred (for debugging)
  bool pre_tactical; // True if the move played leads to opponent having captures
  int outcome;       // Game outcome from side-to-move perspective: -1, 0, +1
};

// Fixed-size buffer for positions within a single game (max ~250 positions per game)
using GamePositions = FixedVector<TrainingPosition, 300>;

// Check if position is quiet (no captures available)
bool is_quiet(const Board& board) {
  MoveList moves;
  generateMoves(board, moves);
  return moves.empty() || !moves[0].isCapture();
}

// Generate a single game and collect training positions
// Returns: outcome from white's perspective at game start (+1 win, 0 draw, -1 loss)
int play_game(RandomBits& rng, search::Searcher& searcher,
              const CompressedTablebaseManager& tb_mgr,
              std::vector<TrainingPosition>& positions,
              int random_plies, std::uint64_t max_nodes) {
  Board board;  // Starting position
  int ply = 0;
  GamePositions game_positions;  // Positions from this game (fixed-size, no heap allocation)

  // Phase 1: Random opening
  while (ply < random_plies) {
    MoveList moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      return (ply % 2 == 0) ? -1 : +1;
    }

    std::uint64_t idx = rng() % moves.size();
    board = makeMove(board, moves[idx]);
    ply++;
  }

  // Phase 2: Play with search until tablebase or game end
  while (true) {
    MoveList moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      int outcome = (ply % 2 == 0) ? -1 : +1;
      for (auto& pos : game_positions) {
        pos.outcome = (pos.ply % 2 == 0) ? outcome : -outcome;
        positions.push_back(pos);
      }
      return outcome;
    }

    // Check for tablebase hit (≤7 pieces)
    int piece_count = std::popcount(board.allPieces());
    if (piece_count <= 7 && board.n_reversible == 0) {
      Value wdl = tb_mgr.lookup_wdl_preloaded(board);
      if (wdl != Value::UNKNOWN) {
        int result;
        if (wdl == Value::WIN) result = +1;
        else if (wdl == Value::LOSS) result = -1;
        else result = 0;

        int outcome = (ply % 2 == 0) ? result : -result;
        for (auto& pos : game_positions) {
          pos.outcome = (pos.ply % 2 == 0) ? outcome : -outcome;
          positions.push_back(pos);
        }
        return outcome;
      }
    }

    // Check for draw by 60 reversible moves
    if (board.n_reversible >= 60) {
      for (auto& pos : game_positions) {
        pos.outcome = 0;
        positions.push_back(pos);
      }
      return 0;
    }

    // Search for best move
    auto result = searcher.search_nodes(board, max_nodes);
    if (result.best_move.from_xor_to == 0) {
      int outcome = (ply % 2 == 0) ? -1 : +1;
      for (auto& pos : game_positions) {
        pos.outcome = (pos.ply % 2 == 0) ? outcome : -outcome;
        positions.push_back(pos);
      }
      return outcome;
    }

    Board next_board = makeMove(board, result.best_move);
    bool pre_tactical = !is_quiet(next_board);

    // Record position if quiet
    if (is_quiet(board)) {
      game_positions.push_back({board, ply, pre_tactical, 0});
    }

    board = next_board;
    ply++;

    if (ply > 500) {
      for (auto& pos : game_positions) {
        pos.outcome = 0;
        positions.push_back(pos);
      }
      return 0;
    }
  }
}

// Write positions to HDF5 file
void write_hdf5(const std::string& filename, const std::vector<TrainingPosition>& positions) {
  using namespace H5;

  const hsize_t n = positions.size();
  if (n == 0) {
    std::cerr << "No positions to write\n";
    return;
  }

  // Prepare data arrays
  std::vector<std::uint32_t> boards(n * 4);
  std::vector<std::uint8_t> pre_tactical(n);
  std::vector<std::int8_t> outcomes(n);

  for (hsize_t i = 0; i < n; ++i) {
    const auto& pos = positions[i];
    boards[i * 4 + 0] = pos.board.white;
    boards[i * 4 + 1] = pos.board.black;
    boards[i * 4 + 2] = pos.board.kings;
    boards[i * 4 + 3] = pos.board.n_reversible;
    pre_tactical[i] = pos.pre_tactical ? 1 : 0;
    outcomes[i] = static_cast<std::int8_t>(pos.outcome);
  }

  // Create HDF5 file
  H5File file(filename, H5F_ACC_TRUNC);

  // Create datasets
  hsize_t board_dims[2] = {n, 4};
  DataSpace board_space(2, board_dims);
  DataSet board_ds = file.createDataSet("boards", PredType::NATIVE_UINT32, board_space);
  board_ds.write(boards.data(), PredType::NATIVE_UINT32);

  hsize_t scalar_dims[1] = {n};
  DataSpace scalar_space(1, scalar_dims);

  DataSet pt_ds = file.createDataSet("pre_tactical", PredType::NATIVE_UINT8, scalar_space);
  pt_ds.write(pre_tactical.data(), PredType::NATIVE_UINT8);

  DataSet out_ds = file.createDataSet("outcomes", PredType::NATIVE_INT8, scalar_space);
  out_ds.write(outcomes.data(), PredType::NATIVE_INT8);

  std::cout << "Wrote " << n << " positions to " << filename << "\n";
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string output_file = "training_data.h5";
  std::string nn_model = "";  // empty = random eval
  std::uint64_t max_nodes = 10000;
  int random_plies = 10;
  std::size_t target_positions = 1000;
  int num_threads = omp_get_max_threads();

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Generate training data for neural network\n\n"
                << "Options:\n"
                << "  -h, --help          Show this help message\n"
                << "  --nodes N           Node limit per move (default: 10000)\n"
                << "  --random-plies N    Random opening moves (default: 10)\n"
                << "  --tb-path PATH      Tablebase directory (default: /home/alvaro/claude/damas)\n"
                << "  --output FILE       Output HDF5 file (default: training_data.h5)\n"
                << "  --positions N       Target number of positions (default: 1000)\n"
                << "  --threads N         Number of threads (default: max available)\n"
                << "  --model FILE        Neural network model for evaluation (default: random)\n";
      return 0;
    } else if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
    } else if (arg == "--nodes" && i + 1 < argc) {
      max_nodes = std::atoll(argv[++i]);
    } else if (arg == "--random-plies" && i + 1 < argc) {
      random_plies = std::atoi(argv[++i]);
    } else if (arg == "--tb-path" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--positions" && i + 1 < argc) {
      target_positions = std::atoll(argv[++i]);
    } else if (arg == "--threads" && i + 1 < argc) {
      num_threads = std::atoi(argv[++i]);
    }
  }

  omp_set_num_threads(num_threads);

  // Seed RNG with nanoseconds
  auto now = std::chrono::high_resolution_clock::now();
  auto base_seed = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          now.time_since_epoch()).count());

  std::cout << "=== Training Data Generator ===\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "RNG base seed: " << base_seed << "\n";
  std::cout << "Random opening plies: " << random_plies << "\n";
  std::cout << "Node limit: " << max_nodes << "\n";
  std::cout << "Evaluation: " << (nn_model.empty() ? "random" : nn_model) << "\n";
  std::cout << "Tablebases: " << tb_dir << "\n";
  std::cout << "Output: " << output_file << "\n";
  std::cout << "Target positions: " << target_positions << "\n\n";

  // Atomic counters for progress tracking
  std::atomic<std::size_t> total_positions{0};
  std::atomic<int> num_games{0};
  std::atomic<int> white_wins{0}, black_wins{0}, draws{0};

  // Shared, preloaded WDL tablebase manager (read-only after preload)
  CompressedTablebaseManager tb_wdl(tb_dir);
  tb_wdl.preload(7);

  // Progress reporting
  std::atomic<double> last_report_time{0.0};

  // Thread-local storage for positions
  std::vector<std::vector<TrainingPosition>> thread_positions(num_threads);

  // Mutex for merging thread-local positions into shared storage
  std::mutex merge_mutex;
  std::vector<TrainingPosition> all_positions;
  // Pre-reserve to avoid reallocations (estimate ~30 positions per game)
  all_positions.reserve(target_positions + target_positions / 10);

  auto start_time = std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();

    // Each thread gets its own RNG and searcher, sharing preloaded WDL tablebases
    RandomBits rng(base_seed + tid * 0x9e3779b97f4a7c15ULL);
    search::Searcher searcher(&tb_wdl, nullptr, 7, 0, nn_model);
    searcher.set_tt_size(32);

    auto& local_positions = thread_positions[tid];

    while (total_positions.load(std::memory_order_relaxed) < target_positions) {
      std::size_t before = local_positions.size();
      int outcome = play_game(rng, searcher, tb_wdl, local_positions, random_plies, max_nodes);
      std::size_t added = local_positions.size() - before;

      // Update atomic counters
      total_positions.fetch_add(added, std::memory_order_relaxed);
      num_games.fetch_add(1, std::memory_order_relaxed);

      if (outcome > 0) white_wins.fetch_add(1, std::memory_order_relaxed);
      else if (outcome < 0) black_wins.fetch_add(1, std::memory_order_relaxed);
      else draws.fetch_add(1, std::memory_order_relaxed);

      // Periodically flush local positions to shared storage to avoid memory pressure
      if (local_positions.size() >= 10000) {
        std::lock_guard<std::mutex> lock(merge_mutex);
        all_positions.insert(all_positions.end(),
                             std::make_move_iterator(local_positions.begin()),
                             std::make_move_iterator(local_positions.end()));
        local_positions.clear();
        local_positions.reserve(10000);  // Keep reasonable capacity
      }

      // Progress report (any thread, every 10 seconds)
      auto now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration<double>(now - start_time).count();
      double last = last_report_time.load(std::memory_order_relaxed);
      if (secs - last >= 10.0) {
        if (last_report_time.compare_exchange_weak(last, secs, std::memory_order_relaxed)) {
          int games = num_games.load(std::memory_order_relaxed);
          std::size_t pos = total_positions.load(std::memory_order_relaxed);
          std::cout << "Games: " << games << "  Positions: " << pos
                    << "  (" << static_cast<int>(pos / secs) << " pos/sec)\n"
                    << std::flush;
        }
      }
    }
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_secs = std::chrono::duration<double>(end_time - start_time).count();

  // Merge remaining thread-local positions
  for (auto& tp : thread_positions) {
    all_positions.insert(all_positions.end(),
                         std::make_move_iterator(tp.begin()),
                         std::make_move_iterator(tp.end()));
    tp.clear();
  }

  // Stats
  std::cout << "\n=== Summary ===\n";
  std::cout << "Time: " << std::fixed << std::setprecision(1) << total_secs << " seconds\n";
  std::cout << "Games: " << num_games.load() << " (W:" << white_wins.load()
            << " B:" << black_wins.load() << " D:" << draws.load() << ")\n";
  std::cout << "Total positions: " << all_positions.size() << "\n";
  std::cout << "Throughput: " << static_cast<int>(all_positions.size() / total_secs) << " pos/sec\n";

  int tactical_count = 0;
  for (const auto& pos : all_positions) {
    if (pos.pre_tactical) tactical_count++;
  }
  std::cout << "Pre-tactical: " << tactical_count << " ("
            << std::fixed << std::setprecision(1)
            << (100.0 * tactical_count / all_positions.size()) << "%)\n";

  // Write to HDF5
  write_hdf5(output_file, all_positions);

  return 0;
}
