// Training data generator for neural network training.
//
// Uses DTM tablebases for search (optimal endgame play) and compressed WDL
// tablebases for adjudication at ≤7 pieces.

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"
#include "tablebase/compression.hpp"
#include "tablebase/tb_probe.hpp"
#include <algorithm>
#include <H5Cpp.h>
#include <omp.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <mutex>
#include <iostream>
#include <iomanip>
#include <unordered_map>
#include <vector>
#include <bit>

// Global stop flag for SIGINT handling
static volatile std::sig_atomic_t g_stop_requested = 0;
static void sigint_handler(int) { g_stop_requested = 1; }

// Training position: board state + metadata
struct TrainingPosition {
  Board board;
  int ply;       // When this position occurred (for debugging)
  int outcome;   // Game outcome from side-to-move perspective: -1, 0, +1
  int score;     // Search score from side-to-move perspective
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
              const CompressedTablebaseManager& wdl_tb,
              std::vector<TrainingPosition>& positions,
              int random_plies, std::uint64_t max_nodes) {
  Board board;  // Starting position
  int ply = 0;
  GamePositions game_positions;  // Positions from this game (fixed-size, no heap allocation)
  std::unordered_map<std::uint64_t, int> position_counts;
  position_counts[board.position_hash()] = 1;

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

    if (++position_counts[board.position_hash()] >= 3) {
      return 0;  // Draw by repetition during random opening
    }
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

    // Adjudicate at ≤7 pieces using WDL tablebases
    int piece_count = std::popcount(board.allPieces());
    if (piece_count <= 7) {
      Value wdl = wdl_tb.lookup_wdl_preloaded(board);
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
    search::TimeControl tc;
    tc.soft_node_limit = max_nodes;
    tc.hard_node_limit = max_nodes * 5;
    auto result = searcher.search(board, 100, tc);
    if (result.best_move.from_xor_to == 0) {
      int outcome = (ply % 2 == 0) ? -1 : +1;
      for (auto& pos : game_positions) {
        pos.outcome = (pos.ply % 2 == 0) ? outcome : -outcome;
        positions.push_back(pos);
      }
      return outcome;
    }

    Board next_board = makeMove(board, result.best_move);

    // Record position if quiet and the resulting position is also quiet
    if (is_quiet(board) && is_quiet(next_board)) {
      game_positions.push_back({board, ply, 0, result.score});
    }

    board = next_board;
    ply++;

    // Check for draw by threefold repetition
    if (++position_counts[board.position_hash()] >= 3) {
      for (auto& pos : game_positions) {
        pos.outcome = 0;
        positions.push_back(pos);
      }
      return 0;
    }

    if (ply > 500) {
      for (auto& pos : game_positions) {
        pos.outcome = 0;
        positions.push_back(pos);
      }
      return 0;
    }
  }
}

// HDF5 writer with extensible datasets for incremental writing
class HDF5Writer {
  H5::H5File file_;
  H5::DataSet board_ds_, outcome_ds_, score_ds_;
  hsize_t count_ = 0;  // Total rows written so far

public:
  explicit HDF5Writer(const std::string& filename) {
    using namespace H5;

    file_ = H5File(filename, H5F_ACC_TRUNC);

    // Create extensible datasets with chunked storage
    hsize_t board_dims[2] = {0, 4};
    hsize_t board_max[2] = {H5S_UNLIMITED, 4};
    hsize_t board_chunk[2] = {4096, 4};
    DataSpace board_space(2, board_dims, board_max);
    DSetCreatPropList board_props;
    board_props.setChunk(2, board_chunk);
    board_ds_ = file_.createDataSet("boards", PredType::NATIVE_UINT32, board_space, board_props);

    hsize_t scalar_dims[1] = {0};
    hsize_t scalar_max[1] = {H5S_UNLIMITED};
    hsize_t scalar_chunk[1] = {4096};
    DataSpace scalar_space(1, scalar_dims, scalar_max);
    DSetCreatPropList scalar_props;
    scalar_props.setChunk(1, scalar_chunk);
    outcome_ds_ = file_.createDataSet("outcomes", PredType::NATIVE_INT8, scalar_space, scalar_props);

    DataSpace score_space(1, scalar_dims, scalar_max);
    DSetCreatPropList score_props;
    score_props.setChunk(1, scalar_chunk);
    score_ds_ = file_.createDataSet("scores", PredType::NATIVE_INT16, score_space, score_props);
  }

  // Append a batch of positions
  void append(const std::vector<TrainingPosition>& positions) {
    using namespace H5;
    if (positions.empty()) return;

    const hsize_t n = positions.size();

    // Prepare data arrays
    std::vector<std::uint32_t> boards(n * 4);
    std::vector<std::int8_t> outcomes(n);
    std::vector<std::int16_t> scores(n);
    for (hsize_t i = 0; i < n; ++i) {
      const auto& pos = positions[i];
      boards[i * 4 + 0] = pos.board.white;
      boards[i * 4 + 1] = pos.board.black;
      boards[i * 4 + 2] = pos.board.kings;
      boards[i * 4 + 3] = pos.board.n_reversible;
      outcomes[i] = static_cast<std::int8_t>(pos.outcome);
      scores[i] = static_cast<std::int16_t>(std::clamp(pos.score, -30000, 30000));
    }

    // Extend datasets
    hsize_t new_count = count_ + n;

    hsize_t board_size[2] = {new_count, 4};
    board_ds_.extend(board_size);
    hsize_t scalar_size[1] = {new_count};
    outcome_ds_.extend(scalar_size);
    score_ds_.extend(scalar_size);

    // Select hyperslab for the new data
    hsize_t board_offset[2] = {count_, 0};
    hsize_t board_count[2] = {n, 4};
    DataSpace board_fspace = board_ds_.getSpace();
    board_fspace.selectHyperslab(H5S_SELECT_SET, board_count, board_offset);
    DataSpace board_mspace(2, board_count);
    board_ds_.write(boards.data(), PredType::NATIVE_UINT32, board_mspace, board_fspace);

    hsize_t scalar_offset[1] = {count_};
    hsize_t scalar_count[1] = {n};
    DataSpace outcome_fspace = outcome_ds_.getSpace();
    outcome_fspace.selectHyperslab(H5S_SELECT_SET, scalar_count, scalar_offset);
    DataSpace scalar_mspace(1, scalar_count);
    outcome_ds_.write(outcomes.data(), PredType::NATIVE_INT8, scalar_mspace, outcome_fspace);

    DataSpace score_fspace = score_ds_.getSpace();
    score_fspace.selectHyperslab(H5S_SELECT_SET, scalar_count, scalar_offset);
    score_ds_.write(scores.data(), PredType::NATIVE_INT16, scalar_mspace, score_fspace);

    count_ = new_count;
    file_.flush(H5F_SCOPE_GLOBAL);
  }

  hsize_t count() const { return count_; }
};

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

  // Install SIGINT handler for graceful shutdown
  std::signal(SIGINT, sigint_handler);

  // Atomic counters for progress tracking
  std::atomic<std::size_t> total_positions{0};
  std::atomic<int> num_games{0};
  std::atomic<int> white_wins{0}, black_wins{0}, draws{0};

  // Compressed WDL tablebases for adjudication at ≤7 pieces
  CompressedTablebaseManager wdl_tb(tb_dir);
  wdl_tb.preload(7);

  // Incremental HDF5 writer
  HDF5Writer writer(output_file);
  std::mutex writer_mutex;

  // Progress reporting
  std::atomic<double> last_report_time{0.0};

  // Thread-local storage for positions
  std::vector<std::vector<TrainingPosition>> thread_positions(num_threads);

  auto start_time = std::chrono::steady_clock::now();

  #pragma omp parallel
  {
    int tid = omp_get_thread_num();

    RandomBits rng(base_seed + tid * 0x9e3779b97f4a7c15ULL);
    search::Searcher searcher("", 0, nn_model);
    searcher.set_tt_size(32);

    auto& local_positions = thread_positions[tid];

    while (total_positions.load(std::memory_order_relaxed) < target_positions
           && !g_stop_requested) {
      std::size_t before = local_positions.size();
      int outcome = play_game(rng, searcher, wdl_tb, local_positions, random_plies, max_nodes);
      std::size_t added = local_positions.size() - before;

      // Update atomic counters
      total_positions.fetch_add(added, std::memory_order_relaxed);
      num_games.fetch_add(1, std::memory_order_relaxed);

      if (outcome > 0) white_wins.fetch_add(1, std::memory_order_relaxed);
      else if (outcome < 0) black_wins.fetch_add(1, std::memory_order_relaxed);
      else draws.fetch_add(1, std::memory_order_relaxed);

      // Flush to HDF5 periodically
      if (local_positions.size() >= 10000) {
        std::lock_guard<std::mutex> lock(writer_mutex);
        writer.append(local_positions);
        local_positions.clear();
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
                    << "  Written: " << writer.count()
                    << "  (" << static_cast<int>(pos / secs) << " pos/sec)\n"
                    << std::flush;
        }
      }
    }

  }

  if (g_stop_requested) {
    std::cout << "\nInterrupted by Ctrl+C, saving collected data...\n";
  }

  auto end_time = std::chrono::steady_clock::now();
  double total_secs = std::chrono::duration<double>(end_time - start_time).count();

  // Flush remaining thread-local positions
  for (auto& tp : thread_positions) {
    if (!tp.empty()) {
      writer.append(tp);
      tp.clear();
    }
  }

  // Stats
  std::cout << "\n=== Summary ===\n";
  std::cout << "Time: " << std::fixed << std::setprecision(1) << total_secs << " seconds\n";
  std::cout << "Games: " << num_games.load() << " (W:" << white_wins.load()
            << " B:" << black_wins.load() << " D:" << draws.load() << ")\n";
  std::cout << "Total positions: " << writer.count() << "\n";
  std::cout << "Throughput: " << static_cast<int>(writer.count() / total_secs) << " pos/sec\n";
  std::cout << "Output: " << output_file << "\n";

  return 0;
}
