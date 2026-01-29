#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tb_probe.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <bit>

// Training position: board state + outcome label
struct TrainingPosition {
  Board board;
  int ply;  // When this position occurred (for debugging)
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
              int random_plies = 10, int search_depth = 8) {
  Board board;  // Starting position
  int ply = 0;

  // Phase 1: Random opening (10 plies)
  while (ply < random_plies) {
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      // Game ended during random phase - loss for side to move
      // Since board is always white-to-move, this is a white loss
      // At even ply, it's actually white's turn from start perspective
      return (ply % 2 == 0) ? -1 : +1;
    }

    // Pick random move
    std::uint64_t idx = rng() % moves.size();
    board = makeMove(board, moves[idx]);
    ply++;
  }

  // Phase 2: Play with search until tablebase or game end
  while (true) {
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      // Loss for side to move (white in our representation)
      return (ply % 2 == 0) ? -1 : +1;
    }

    // Check for tablebase hit (≤7 pieces)
    int piece_count = std::popcount(board.allPieces());
    if (piece_count <= 7 && board.n_reversible == 0) {
      // Use tablebase result
      tablebase::DTM dtm = dtm_mgr->lookup_dtm(board);
      if (dtm != tablebase::DTM_UNKNOWN) {
        int result;
        if (dtm > 0) result = +1;       // White wins
        else if (dtm < 0) result = -1;  // White loses
        else result = 0;                 // Draw

        // Adjust for perspective: at even ply, white-to-move = original white
        return (ply % 2 == 0) ? result : -result;
      }
    }

    // Check for draw by 60 reversible moves
    if (board.n_reversible >= 60) {
      return 0;
    }

    // Record position if quiet
    if (is_quiet(board)) {
      positions.push_back({board, ply});
    }

    // Search for best move
    auto result = searcher.search(board, search_depth);
    if (result.best_move.from_xor_to == 0) {
      // No move found (shouldn't happen if moves is non-empty)
      return (ply % 2 == 0) ? -1 : +1;
    }

    board = makeMove(board, result.best_move);
    ply++;

    // Safety limit
    if (ply > 500) {
      return 0;  // Draw by excessive length
    }
  }
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  int search_depth = 8;
  int random_plies = 10;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth" && i + 1 < argc) {
      search_depth = std::atoi(argv[++i]);
    } else if (arg == "--random-plies" && i + 1 < argc) {
      random_plies = std::atoi(argv[++i]);
    } else if (arg == "--tb-path" && i + 1 < argc) {
      tb_dir = argv[++i];
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
  std::cout << "Tablebases: " << tb_dir << "\n\n";

  // Create searcher and DTM manager
  search::Searcher searcher(tb_dir, 7, 6);
  tablebase::DTMTablebaseManager dtm_mgr(tb_dir);

  // Play one game
  std::vector<TrainingPosition> positions;
  int outcome = play_game(rng, searcher, &dtm_mgr, positions, random_plies, search_depth);

  // Display results
  std::cout << "Game outcome: ";
  if (outcome > 0) std::cout << "WHITE WINS";
  else if (outcome < 0) std::cout << "BLACK WINS";
  else std::cout << "DRAW";
  std::cout << "\n\n";

  std::cout << "Collected " << positions.size() << " training positions:\n";
  std::cout << std::string(60, '-') << "\n";

  for (size_t i = 0; i < positions.size(); ++i) {
    const auto& pos = positions[i];

    // Determine label: outcome from the perspective of side to move at this position
    // pos.ply is the ply when this position occurred
    // At even ply, it's original white's turn; at odd ply, original black's turn
    int label = (pos.ply % 2 == 0) ? outcome : -outcome;

    std::cout << "Position " << std::setw(3) << i << " (ply " << std::setw(3) << pos.ply << ")";
    std::cout << "  Label: " << std::setw(2) << label;
    std::cout << " (" << (label > 0 ? "win" : (label < 0 ? "loss" : "draw")) << ")\n";
    std::cout << pos.board;
    std::cout << "Pieces: " << std::popcount(pos.board.allPieces()) << "\n\n";
  }

  return 0;
}
