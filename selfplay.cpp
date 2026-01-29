#include "core/board.h"
#include "core/movegen.h"
#include "core/notation.h"
#include "search/search.h"
#include "tablebase/tb_probe.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <bit>

// DTM manager for endgame display
static std::unique_ptr<tablebase::DTMTablebaseManager> g_dtm_manager;

// Material string like "KKvK" or "PPvK"
std::string materialString(const Material& m) {
  std::string result;
  for (int i = 0; i < m.white_queens; ++i) result += 'K';
  for (int i = 0; i < m.back_white_pawns + m.other_white_pawns; ++i) result += 'P';
  result += 'v';
  for (int i = 0; i < m.black_queens; ++i) result += 'K';
  for (int i = 0; i < m.back_black_pawns + m.other_black_pawns; ++i) result += 'P';
  return result;
}

// DTM description like "winning in 5 moves"
std::string dtmDescription(tablebase::DTM d) {
  if (d == tablebase::DTM_UNKNOWN) return "unknown";
  if (d == tablebase::DTM_DRAW) return "draw";
  if (d > 0) return "winning in " + std::to_string(d) + " moves";
  if (d < 0) return "losing in " + std::to_string(-d) + " moves";
  return "draw";
}

// Print pieces with positions (from display perspective)
void printPieces(const Board& b) {
  std::cout << "  White: ";
  for (int sq = 0; sq < 32; ++sq) {
    if (b.white & (1u << sq)) {
      std::cout << (b.kings & (1u << sq) ? "K" : "P") << (sq + 1) << " ";
    }
  }
  std::cout << "\n  Black: ";
  for (int sq = 0; sq < 32; ++sq) {
    if (b.black & (1u << sq)) {
      std::cout << (b.kings & (1u << sq) ? "K" : "P") << (sq + 1) << " ";
    }
  }
  std::cout << "\n";
}

// Find the FullMove matching a compact Move
FullMove findFullMove(const Board& board, const Move& move) {
  std::vector<FullMove> fullMoves;
  generateFullMoves(board, fullMoves);
  for (const auto& fm : fullMoves) {
    if (fm.move.from_xor_to == move.from_xor_to &&
        fm.move.captures == move.captures) {
      return fm;
    }
  }
  // Fallback: return empty FullMove
  return FullMove{};
}

// Flip a square for black's perspective
inline int flipSquare(int sq) { return 31 - sq; }

// Convert FullMove to string with perspective handling
std::string moveToStringPerspective(const FullMove& fullMove, bool blackPerspective) {
  const auto& path = fullMove.path;
  if (path.empty()) return "?";

  std::ostringstream oss;
  char sep = fullMove.move.isCapture() ? 'x' : '-';

  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0) oss << sep;
    int sq = path[i];
    if (blackPerspective) sq = flipSquare(sq);
    oss << (sq + 1);
  }
  return oss.str();
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

  // Search and print PV with proper perspective
  // ply: current ply (0 = white's first move, 1 = black's first move, etc.)
  search::SearchResult search(const Board& board, int max_depth, int ply) {
    search::SearchResult best_result;

    for (int depth = 1; depth <= max_depth; ++depth) {
      auto result = searcher_.search(board, depth);
      best_result = result;
      best_result.depth = depth;

      // Print search info
      std::cout << "  depth " << std::setw(2) << depth
                << "  score " << std::setw(6) << result.score
                << "  nodes " << std::setw(8) << result.nodes;

      // Print PV with full path notation
      std::cout << "  pv";
      Board pos = board;
      Move pv_move = result.best_move;
      int pv_ply = ply;

      // First move in PV
      if (pv_move.from_xor_to != 0) {
        FullMove fm = findFullMove(pos, pv_move);
        std::cout << " " << moveToStringPerspective(fm, pv_ply % 2 == 1);
      }

      // Extract more of the PV by following TT
      for (int i = 0; i < depth - 1 && pv_move.from_xor_to != 0; ++i) {
        pos = makeMove(pos, pv_move);
        pv_ply++;
        auto child_result = searcher_.search(pos, 1);
        if (child_result.best_move.from_xor_to == 0) break;
        FullMove fm = findFullMove(pos, child_result.best_move);
        std::cout << " " << moveToStringPerspective(fm, pv_ply % 2 == 1);
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
  bool use_hash_eval = false;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--depth" && i + 1 < argc) {
      max_depth = std::atoi(argv[++i]);
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

  // Initialize DTM manager for endgame display
  if (!tb_dir.empty()) {
    g_dtm_manager = std::make_unique<tablebase::DTMTablebaseManager>(tb_dir);
  }

  Board board;  // Initial position
  std::vector<FullMove> game_moves;  // Store full moves for notation
  int ply = 0;  // 0 = white's first move, 1 = black's first, etc.

  // Show initial position
  std::cout << "Initial position:\n" << board;

  while (true) {
    // Check for draw by reversible move rule
    if (board.n_reversible >= 60) {
      std::cout << "\n*** GAME OVER ***\n";
      std::cout << "Draw by 60 reversible moves\n";
      break;
    }

    // Generate moves
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      // Game over - current side to move loses
      std::cout << "\n*** GAME OVER ***\n";
      bool white_to_move = (ply % 2 == 0);
      if (white_to_move) {
        std::cout << "Black wins! (White has no moves)\n";
      } else {
        std::cout << "White wins! (Black has no moves)\n";
      }
      break;
    }

    bool white_to_move = (ply % 2 == 0);
    std::cout << "\n--- Move " << (ply + 1) << " ---\n";
    std::cout << (white_to_move ? "White" : "Black") << " to move\n";
    std::cout << "Searching...\n";

    auto result = searcher.search(board, max_depth, ply);

    // Find full move for proper notation
    FullMove bestFullMove = findFullMove(board, result.best_move);
    std::string move_str = moveToStringPerspective(bestFullMove, !white_to_move);
    std::cout << "\nBest move: " << move_str;
    if (result.best_move.isCapture()) {
      std::cout << " (captures " << std::popcount(result.best_move.captures) << ")";
    }
    std::cout << "\n";

    // Record move
    game_moves.push_back(bestFullMove);

    // Make the move
    board = makeMove(board, result.best_move);
    ply++;

    // Show position (flip if black's turn so display matches real game)
    bool next_is_white = (ply % 2 == 0);
    Board display = next_is_white ? board : flip(board);
    std::cout << "\nPosition after move " << ply
              << " (" << (next_is_white ? "White" : "Black") << " to move):\n"
              << display;

    // Print material count
    int white_pieces = std::popcount(board.white);
    int black_pieces = std::popcount(board.black);
    // Remember: board is flipped, so if it's black's turn next,
    // what's stored as "white" is actually black's pieces
    if (!next_is_white) {
      std::swap(white_pieces, black_pieces);
    }
    int total_pieces = white_pieces + black_pieces;

    // For endgame positions (≤6 pieces), show DTM-style output
    if (total_pieces <= 6 && g_dtm_manager) {
      Material m = get_material(board);
      // Flip material if it's black's turn (internal board is flipped)
      if (!next_is_white) {
        m = flip(m);
      }
      std::cout << "Material: " << materialString(m) << " (" << total_pieces << " pieces)\n";
      printPieces(display);

      tablebase::DTM dtm = g_dtm_manager->lookup_dtm(board);
      std::cout << "Position: " << dtmDescription(dtm) << "\n";
    } else {
      std::cout << "Material: White " << white_pieces << ", Black " << black_pieces << "\n";
    }

    // Check for decisive result
    if (search::is_mate_score(result.score)) {
      std::cout << "\n*** GAME OVER ***\n";
      if (result.score > 0) {
        std::cout << (next_is_white ? "Black" : "White") << " wins!\n";
      } else {
        std::cout << (next_is_white ? "White" : "Black") << " wins!\n";
      }
      break;
    }
  }

  // Print move list using gameToString
  std::cout << "\n=== Move List ===\n";
  std::cout << gameToString(game_moves) << "\n";

  return 0;
}
