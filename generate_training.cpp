#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tb_probe.hpp"
#include <H5Cpp.h>
#include <chrono>
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

// Check if position is quiet (no captures available)
bool is_quiet(const Board& board) {
  std::vector<Move> moves;
  generateMoves(board, moves);
  return moves.empty() || !moves[0].isCapture();
}

// Generate a single game and collect training positions
// Returns: outcome from white's perspective at game start (+1 win, 0 draw, -1 loss)
int play_game(RandomBits& rng, search::Searcher& searcher,
              tablebase::DTMTablebaseManager* dtm_mgr,
              std::vector<TrainingPosition>& positions,
              int random_plies, int search_depth) {
  Board board;  // Starting position
  int ply = 0;
  std::vector<TrainingPosition> game_positions;  // Positions from this game

  // Phase 1: Random opening
  while (ply < random_plies) {
    std::vector<Move> moves;
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
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      int outcome = (ply % 2 == 0) ? -1 : +1;
      // Label all positions from this game
      for (auto& pos : game_positions) {
        pos.outcome = (pos.ply % 2 == 0) ? outcome : -outcome;
        positions.push_back(pos);
      }
      return outcome;
    }

    // Check for tablebase hit (≤7 pieces)
    int piece_count = std::popcount(board.allPieces());
    if (piece_count <= 7 && board.n_reversible == 0) {
      tablebase::DTM dtm = dtm_mgr->lookup_dtm(board);
      if (dtm != tablebase::DTM_UNKNOWN) {
        int result;
        if (dtm > 0) result = +1;
        else if (dtm < 0) result = -1;
        else result = 0;

        int outcome = (ply % 2 == 0) ? result : -result;
        // Label all positions from this game
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
    auto result = searcher.search(board, search_depth);
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
  int search_depth = 8;
  int random_plies = 10;
  int num_games = 1;
  bool verbose = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth" && i + 1 < argc) {
      search_depth = std::atoi(argv[++i]);
    } else if (arg == "--random-plies" && i + 1 < argc) {
      random_plies = std::atoi(argv[++i]);
    } else if (arg == "--tb-path" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--games" && i + 1 < argc) {
      num_games = std::atoi(argv[++i]);
    } else if (arg == "--verbose" || arg == "-v") {
      verbose = true;
    }
  }

  // Seed RNG with nanoseconds
  auto now = std::chrono::high_resolution_clock::now();
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      now.time_since_epoch()).count();
  RandomBits rng(static_cast<std::uint64_t>(ns));

  std::cout << "=== Training Data Generator ===\n";
  std::cout << "RNG seed (ns): " << ns << "\n";
  std::cout << "Random opening plies: " << random_plies << "\n";
  std::cout << "Search depth: " << search_depth << "\n";
  std::cout << "Tablebases: " << tb_dir << "\n";
  std::cout << "Output: " << output_file << "\n";
  std::cout << "Games: " << num_games << "\n\n";

  // Create searcher and DTM manager
  search::Searcher searcher(tb_dir, 7, 6);
  tablebase::DTMTablebaseManager dtm_mgr(tb_dir);

  // Play games and collect positions
  std::vector<TrainingPosition> all_positions;
  int white_wins = 0, black_wins = 0, draws = 0;

  for (int game = 0; game < num_games; ++game) {
    std::vector<TrainingPosition> positions;
    int outcome = play_game(rng, searcher, &dtm_mgr, all_positions, random_plies, search_depth);

    if (outcome > 0) white_wins++;
    else if (outcome < 0) black_wins++;
    else draws++;

    if (verbose || num_games == 1) {
      std::cout << "Game " << (game + 1) << ": ";
      if (outcome > 0) std::cout << "WHITE WINS";
      else if (outcome < 0) std::cout << "BLACK WINS";
      else std::cout << "DRAW";
      std::cout << " (" << all_positions.size() << " total positions)\n";
    } else if ((game + 1) % 100 == 0) {
      std::cout << "Completed " << (game + 1) << " games, "
                << all_positions.size() << " positions\n";
    }
  }

  // Stats
  std::cout << "\n=== Summary ===\n";
  std::cout << "Games: " << num_games << " (W:" << white_wins
            << " B:" << black_wins << " D:" << draws << ")\n";
  std::cout << "Total positions: " << all_positions.size() << "\n";

  int tactical_count = 0;
  for (const auto& pos : all_positions) {
    if (pos.pre_tactical) tactical_count++;
  }
  std::cout << "Pre-tactical: " << tactical_count << " ("
            << (100.0 * tactical_count / all_positions.size()) << "%)\n";

  // Write to HDF5
  write_hdf5(output_file, all_positions);

  // If verbose and single game, print positions
  if (verbose && num_games == 1) {
    std::cout << "\nPositions:\n" << std::string(60, '-') << "\n";
    for (size_t i = 0; i < all_positions.size(); ++i) {
      const auto& pos = all_positions[i];
      std::cout << "Position " << std::setw(3) << i
                << " (ply " << std::setw(3) << pos.ply << ")";
      std::cout << "  Label: " << std::setw(2) << pos.outcome;
      std::cout << " (" << (pos.outcome > 0 ? "win" : (pos.outcome < 0 ? "loss" : "draw")) << ")";
      if (pos.pre_tactical) std::cout << "  [PRE-TACTICAL]";
      std::cout << "\n" << pos.board;
      std::cout << "Pieces: " << std::popcount(pos.board.allPieces()) << "\n\n";
    }
  }

  return 0;
}
