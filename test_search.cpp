#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include <iostream>
#include <chrono>
#include <bit>

// Helper to print a move
void print_move(const Move& m, const Board& board) {
  Bb from = m.from_xor_to & board.white;
  Bb to = m.from_xor_to & ~board.white;
  if (from == 0) {
    // XOR produced 0 for one side - the move is to/from the same bit position
    from = to = m.from_xor_to;
  }
  int from_sq = std::countr_zero(from) + 1;  // 1-indexed
  int to_sq = std::countr_zero(to) + 1;

  std::cout << from_sq;
  if (m.isCapture()) {
    std::cout << "x" << to_sq << " (captures " << std::popcount(m.captures) << ")";
  } else {
    std::cout << "-" << to_sq;
  }
}

// Test basic search on initial position
void test_initial_position(search::Searcher& searcher) {
  std::cout << "\n=== Initial Position ===\n";
  Board board;
  std::cout << board << "\n";

  for (int depth = 1; depth <= 8; ++depth) {
    auto start = std::chrono::high_resolution_clock::now();
    auto result = searcher.search(board, depth);
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double nps = ms > 0 ? (result.nodes * 1000.0 / ms) : 0;

    std::cout << "depth " << depth << ": score " << result.score << ", best move ";
    print_move(result.best_move, board);
    std::cout << ", " << result.nodes << " nodes";
    std::cout << " [" << ms << " ms, " << static_cast<int>(nps / 1000) << " knps";
    if (result.tb_hits > 0) {
      std::cout << ", " << result.tb_hits << " TB hits";
    }
    std::cout << "]\n";
  }
}

// Test tablebase positions (simple endgames)
void test_tablebase_position(search::Searcher& searcher) {
  std::cout << "\n=== Tablebase Position (1v1) ===\n";

  // Simple 1v1: White queen on square 17, Black pawn on square 28
  // White queen should win
  Board board(0, 0, 0);
  board.white = 1u << 17;  // White queen on center
  board.black = 1u << 28;  // Black pawn on back rank
  board.kings = board.white;  // White piece is a queen

  std::cout << board << "\n";
  std::cout << "Pieces: " << std::popcount(board.allPieces()) << "\n";

  auto result = searcher.search_iterative(board, 20);
  std::cout << "Result: score " << result.score << ", depth " << result.depth;
  if (result.best_move.from_xor_to != 0) {
    std::cout << ", best move ";
    print_move(result.best_move, board);
  }
  std::cout << "\n";
  std::cout << "TB hits: " << result.tb_hits << "\n";
}

// Test a position with forced captures
void test_capture_position() {
  std::cout << "\n=== Forced Capture Position ===\n";

  // White pawn at 8 can capture black pawn at 12, landing on 16
  // (moveNW(8) = 12, moveNW(12) = 16)
  Board board(0, 0, 0);
  board.white = 1u << 8;   // White pawn at square 8
  board.black = 1u << 12;  // Black pawn at square 12 (capturable)

  std::cout << board << "\n";

  std::vector<Move> moves;
  generateMoves(board, moves);

  std::cout << "Legal moves: " << moves.size() << "\n";
  for (const auto& m : moves) {
    std::cout << "  ";
    print_move(m, board);
    std::cout << "\n";
  }

  // Verify it's a capture
  if (!moves.empty() && moves[0].isCapture()) {
    std::cout << "Capture forced: YES\n";
  }
}

// Test quiescence search - verify captures aren't treated as leaves
void test_quiescence(search::Searcher& searcher) {
  std::cout << "\n=== Quiescence Search Test ===\n";

  // Position where white has a capture available
  // The search should resolve the capture, not just evaluate the position
  Board board(0, 0, 0);
  board.white = 1u << 8;   // White pawn at 8
  board.black = 1u << 12;  // Black pawn at 12 (can be captured)

  std::cout << board << "\n";
  std::cout << "White to move, capture available\n";

  // Search with depth 1 - should still search through the capture
  auto result = searcher.search(board, 1);
  std::cout << "Depth 1 result: score " << result.score;
  std::cout << ", nodes " << result.nodes;
  if (result.best_move.isCapture()) {
    std::cout << ", best move is a capture";
  }
  std::cout << "\n";

  // The score should reflect the capture (material gain)
  if (result.score > 0) {
    std::cout << "Quiescence working: score reflects material after capture\n";
  }
}

// Test iterative deepening
void test_iterative_deepening(search::Searcher& searcher) {
  std::cout << "\n=== Iterative Deepening ===\n";
  Board board;

  auto start = std::chrono::high_resolution_clock::now();
  auto result = searcher.search_iterative(board, 10);
  auto end = std::chrono::high_resolution_clock::now();

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "Reached depth " << result.depth << "\n";
  std::cout << "Score: " << result.score << "\n";
  std::cout << "Best move: ";
  print_move(result.best_move, board);
  std::cout << "\n";
  std::cout << "Nodes: " << result.nodes << "\n";
  std::cout << "Time: " << ms << " ms\n";
  if (result.tb_hits > 0) {
    std::cout << "TB hits: " << result.tb_hits << "\n";
  }
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  int tb_limit = 7;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--tb-dir" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--tb-limit" && i + 1 < argc) {
      tb_limit = std::atoi(argv[++i]);
    } else if (arg == "--no-tb") {
      tb_dir = "";
    }
  }

  std::cout << "Torquemada Search Engine Test\n";
  std::cout << "=============================\n";

  if (!tb_dir.empty()) {
    std::cout << "Tablebase directory: " << tb_dir << "\n";
    std::cout << "Tablebase piece limit: " << tb_limit << "\n";
  } else {
    std::cout << "Tablebases disabled\n";
  }

  search::Searcher searcher(tb_dir, tb_limit);
  searcher.set_tt_size(128);  // 128 MB TT

  test_capture_position();
  test_quiescence(searcher);
  test_initial_position(searcher);
  test_tablebase_position(searcher);
  test_iterative_deepening(searcher);

  return 0;
}
