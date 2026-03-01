#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include "tablebase/compression.hpp"
#include "nn/mlp.hpp"
#include <iostream>
#include <bit>
#include <optional>
#include <cstdio>

// Test position: wp:15,24 wq:26 bp:9,31 bq:10 (1-indexed human squares)
// Convert to 0-indexed: wp:14,23 wq:25 bp:8,30 bq:9
// White to move. 6 pieces. WDL should be probed.

int main() {
  // Build the position (white to move, so stored directly)
  // Human squares are 1-32, internal bits are 0-31
  Board board(0, 0, 0);
  board.white = (1u << 14) | (1u << 23) | (1u << 25);  // wp:15,24 wq:26 (1-indexed)
  board.black = (1u << 8) | (1u << 30) | (1u << 9);     // bp:9,31 bq:10 (1-indexed)
  board.kings = (1u << 25) | (1u << 9);                   // wq:26, bq:10 (1-indexed)

  std::cout << "Position:\n" << board << "\n";
  std::cout << "Piece count: " << std::popcount(board.allPieces()) << "\n";
  std::cout << "White: 0x" << std::hex << board.white << std::dec << "\n";
  std::cout << "Black: 0x" << std::hex << board.black << std::dec << "\n";
  std::cout << "Kings: 0x" << std::hex << board.kings << std::dec << "\n\n";

  // Show legal moves
  MoveList moves;
  generateMoves(board, moves);
  std::cout << "Legal moves: " << moves.size() << "\n";
  for (const Move& m : moves) {
    Bb from_bit = m.from_xor_to & board.white;
    Bb to_bit = m.from_xor_to & ~board.white;
    if (from_bit == 0) {
      from_bit = to_bit = m.from_xor_to;
    }
    int from_sq = std::countr_zero(from_bit) + 1;
    int to_sq = std::countr_zero(to_bit) + 1;
    std::cout << "  " << from_sq << (m.isCapture() ? "x" : "-") << to_sq << "\n";
  }
  std::cout << "\n";

  // Load NN model
  std::string nn_model = "/home/alvaro/claude/torquemada-gh-pages/models/model_006.bin";
  nn::MLP model(nn_model);
  std::cout << "NN model loaded\n";

  // Evaluate the root position with NN
  int root_eval = model.evaluate(board);
  float p_loss, p_draw, p_win;
  model.predict_proba(board, p_loss, p_draw, p_win);
  std::cout << "Root NN eval: " << root_eval
            << " (p_win=" << p_win << " p_draw=" << p_draw << " p_loss=" << p_loss << ")\n";

  // Evaluate each child position
  std::cout << "\nChild position evaluations:\n";
  for (const Move& m : moves) {
    Board child = makeMove(board, m);
    int child_eval = model.evaluate(child);
    float cp_loss, cp_draw, cp_win;
    model.predict_proba(child, cp_loss, cp_draw, cp_win);

    Bb from_bit = m.from_xor_to & board.white;
    Bb to_bit = m.from_xor_to & ~board.white;
    if (from_bit == 0) {
      from_bit = to_bit = m.from_xor_to;
    }
    int from_sq = std::countr_zero(from_bit) + 1;
    int to_sq = std::countr_zero(to_bit) + 1;

    std::cout << "  " << from_sq << (m.isCapture() ? "x" : "-") << to_sq
              << ": eval=" << child_eval
              << " (p_win=" << cp_win << " p_draw=" << cp_draw << " p_loss=" << cp_loss << ")"
              << " [negated=" << -child_eval << "]\n";
  }

  // Now load CWDL tablebases and set up the searcher
  std::cout << "\n=== Search with WDL tablebases ===\n";
  CompressedTablebaseManager cwdl_manager("/home/alvaro/claude/damas");
  cwdl_manager.preload(7);

  // Probe the root position's WDL directly
  Value root_wdl = cwdl_manager.lookup_wdl(board);
  std::cout << "Root WDL: " << static_cast<int>(root_wdl) << " (0=DRAW, 1=WIN, 2=LOSS, 3=UNKNOWN)\n";

  // Probe each child's WDL
  for (const Move& m : moves) {
    Board child = makeMove(board, m);
    Value child_wdl = cwdl_manager.lookup_wdl(child);

    Bb from_bit = m.from_xor_to & board.white;
    Bb to_bit = m.from_xor_to & ~board.white;
    if (from_bit == 0) {
      from_bit = to_bit = m.from_xor_to;
    }
    int from_sq = std::countr_zero(from_bit) + 1;
    int to_sq = std::countr_zero(to_bit) + 1;

    std::cout << "  " << from_sq << (m.isCapture() ? "x" : "-") << to_sq
              << ": WDL=" << static_cast<int>(child_wdl) << "\n";
  }

  // Set up searcher with DTM (5-piece) and WDL (7-piece)
  search::Searcher searcher("/home/alvaro/claude/damas", 5, nn_model);
  searcher.set_verbose(true);

  // Set up WDL probe
  searcher.set_wdl_probe([&cwdl_manager](const Board& b) -> std::optional<int> {
    Value v = cwdl_manager.lookup_wdl(b);
    switch (v) {
      case Value::WIN: return 1;
      case Value::DRAW: return 0;
      case Value::LOSS: return -1;
      default: return std::nullopt;
    }
  }, 7);

  // Search at various depths
  for (int depth = 1; depth <= 12; ++depth) {
    auto result = searcher.search(board, depth);
    // Format score like the web UI
    int score = result.score;
    char score_str[64];
    if (std::abs(score) > 29000) {
      int mate_in = (30000 - std::abs(score) + 1) / 2;
      std::snprintf(score_str, sizeof(score_str), "%sM%d",
                    score > 0 ? "" : "-", mate_in);
    } else if (std::abs(score) <= 10000) {
      double val = score / 100.0;
      std::snprintf(score_str, sizeof(score_str), "%+.2f(tablas)", val);
    } else {
      int raw = score > 0 ? score - 10000 : score + 10000;
      double val = raw / 100.0;
      std::snprintf(score_str, sizeof(score_str), "%+.2f", val);
    }
    std::cout << "  FINAL depth " << depth << ": " << score_str
              << " (raw=" << score << ") nodes=" << result.nodes
              << " tb_hits=" << result.tb_hits << "\n\n";
  }

  return 0;
}
