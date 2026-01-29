#include "core/board.h"
#include "core/movegen.h"
#include "search/search.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <bit>

// Convert square index to algebraic notation (1-32)
std::string sq_to_str(int sq) {
  return std::to_string(sq + 1);
}

// Get from/to squares from a move
std::pair<int, int> get_from_to(const Move& m, Bb white) {
  Bb from_bb = m.from_xor_to & white;
  Bb to_bb = m.from_xor_to & ~white;

  // Handle edge case where piece moves within same bitboard region
  if (from_bb == 0 || to_bb == 0) {
    // Both bits are in from_xor_to, need to figure out which is which
    int bit1 = std::countr_zero(m.from_xor_to);
    int bit2 = std::countr_zero(m.from_xor_to & ~(1u << bit1));
    // The 'from' square has the piece
    if ((1u << bit1) & white) {
      return {bit1, bit2};
    } else {
      return {bit2, bit1};
    }
  }

  return {std::countr_zero(from_bb), std::countr_zero(to_bb)};
}

// Format a move as string
std::string move_to_str(const Move& m, Bb white) {
  auto [from, to] = get_from_to(m, white);
  std::string s = sq_to_str(from);
  s += m.isCapture() ? "x" : "-";
  s += sq_to_str(to);
  return s;
}

// Print board from a specific side's perspective
void print_board(const Board& board, bool white_perspective, int move_number) {
  // If we need black's perspective, flip the board for display
  Board display = white_perspective ? board : flip(board);

  std::cout << "\n";
  if (white_perspective) {
    std::cout << "Position after move " << move_number << " (White to move):\n";
  } else {
    std::cout << "Position after move " << move_number << " (Black to move):\n";
  }

  // Print with rank/file guides
  std::cout << "  +---------------+\n";
  for (int row = 7; row >= 0; --row) {
    std::cout << (row + 1) << " |";
    if (row % 2 == 0) std::cout << " ";
    for (int col = 3; col >= 0; --col) {
      int bit = row * 4 + col;
      char piece = '.';
      if ((display.white >> bit) & 1) {
        piece = ((display.kings >> bit) & 1) ? 'W' : 'w';
      } else if ((display.black >> bit) & 1) {
        piece = ((display.kings >> bit) & 1) ? 'B' : 'b';
      }
      std::cout << piece << " ";
    }
    if (row % 2 == 1) std::cout << " ";
    std::cout << "|\n";
  }
  std::cout << "  +---------------+\n";
  std::cout << "    a b c d e f g h\n";
}

// Verbose searcher that prints info after each depth
class VerboseSearcher {
public:
  VerboseSearcher(const std::string& tb_dir, int tb_limit, bool use_hash_eval = false)
      : searcher_(tb_dir, tb_limit) {
    searcher_.set_tt_size(128);
    if (use_hash_eval) {
      searcher_.set_eval(search::hash_eval);
    }
  }

  search::SearchResult search(const Board& board, int max_depth) {
    search::SearchResult best_result;

    for (int depth = 1; depth <= max_depth; ++depth) {
      auto result = searcher_.search(board, depth);
      best_result = result;
      best_result.depth = depth;

      // Print search info
      std::cout << "  depth " << std::setw(2) << depth
                << "  score " << std::setw(6) << result.score
                << "  nodes " << std::setw(8) << result.nodes;

      // Print PV (just best move for now, full PV would need TT extraction)
      std::cout << "  pv " << move_to_str(result.best_move, board.white);

      // Extract more of the PV by following TT
      Board pos = board;
      Move pv_move = result.best_move;
      for (int i = 0; i < depth - 1 && pv_move.from_xor_to != 0; ++i) {
        pos = makeMove(pos, pv_move);
        auto child_result = searcher_.search(pos, 1);
        if (child_result.best_move.from_xor_to == 0) break;
        std::cout << " " << move_to_str(child_result.best_move, pos.white);
        pv_move = child_result.best_move;
      }

      std::cout << "\n";

      // Early exit on mate
      if (search::is_mate_score(result.score)) {
        break;
      }
    }

    return best_result;
  }

  void clear_tt() { searcher_.clear_tt(); }

private:
  search::Searcher searcher_;
};

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  int tb_limit = 7;
  int max_depth = 8;
  int max_moves = 100;
  bool use_hash_eval = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth" && i + 1 < argc) {
      max_depth = std::atoi(argv[++i]);
    } else if (arg == "--moves" && i + 1 < argc) {
      max_moves = std::atoi(argv[++i]);
    } else if (arg == "--no-tb") {
      tb_dir = "";
    } else if (arg == "--hash-eval") {
      use_hash_eval = true;
    }
  }

  std::cout << "=== Torquemada Self-Play Game ===\n";
  std::cout << "Search depth: " << max_depth << "\n";
  std::cout << "Tablebases: " << (tb_dir.empty() ? "disabled" : tb_dir) << "\n";
  std::cout << "Evaluation: " << (use_hash_eval ? "hash-based" : "material+positional") << "\n";
  std::cout << "\n";

  VerboseSearcher searcher(tb_dir, tb_limit, use_hash_eval);

  Board board;  // Initial position
  std::vector<std::string> move_list;
  int move_number = 0;
  bool white_to_move_in_reality = true;  // Track actual side to move

  // Show initial position
  std::cout << "Initial position:\n";
  print_board(board, true, 0);

  while (move_number < max_moves) {
    // Generate moves
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      // Game over - current side to move loses
      std::cout << "\n*** GAME OVER ***\n";
      if (white_to_move_in_reality) {
        std::cout << "Black wins! (White has no moves)\n";
      } else {
        std::cout << "White wins! (Black has no moves)\n";
      }
      break;
    }

    // Check for draw by 25-move rule (50 reversible half-moves)
    if (board.n_reversible >= 50) {
      std::cout << "\n*** GAME OVER ***\n";
      std::cout << "Draw by 50-move rule (25 reversible moves per side)\n";
      break;
    }

    move_number++;

    std::cout << "\n--- Move " << move_number << " ---\n";
    std::cout << (white_to_move_in_reality ? "White" : "Black") << " to move\n";
    std::cout << "Searching...\n";

    auto result = searcher.search(board, max_depth);

    std::string move_str = move_to_str(result.best_move, board.white);
    std::cout << "\nBest move: " << move_str;
    if (result.best_move.isCapture()) {
      std::cout << " (captures " << std::popcount(result.best_move.captures) << ")";
    }
    std::cout << "\n";

    // Record move
    move_list.push_back(move_str);

    // Make the move
    board = makeMove(board, result.best_move);
    white_to_move_in_reality = !white_to_move_in_reality;

    // Show position (from white's real perspective)
    // Since board is always stored as white-to-move, we need to flip for display
    // when it's actually black's turn in the real game
    print_board(board, white_to_move_in_reality, move_number);

    // Print material count
    int white_pieces = std::popcount(board.white);
    int black_pieces = std::popcount(board.black);
    // Remember: board is flipped, so if it's black's turn in reality,
    // what's stored as "white" is actually black's pieces
    if (!white_to_move_in_reality) {
      std::swap(white_pieces, black_pieces);
    }
    std::cout << "Material: White " << white_pieces << ", Black " << black_pieces << "\n";

    // Check for decisive result
    if (search::is_mate_score(result.score)) {
      std::cout << "\n*** GAME OVER ***\n";
      if (result.score > 0) {
        std::cout << (white_to_move_in_reality ? "Black" : "White") << " wins!\n";
      } else {
        std::cout << (white_to_move_in_reality ? "White" : "Black") << " wins!\n";
      }
      break;
    }
  }

  // Print move list
  std::cout << "\n=== Move List ===\n";
  for (size_t i = 0; i < move_list.size(); ++i) {
    if (i % 2 == 0) {
      std::cout << ((i / 2) + 1) << ". ";
    }
    std::cout << move_list[i];
    if (i % 2 == 1) {
      std::cout << "\n";
    } else {
      std::cout << " ";
    }
  }
  if (move_list.size() % 2 == 1) {
    std::cout << "\n";
  }

  return 0;
}
