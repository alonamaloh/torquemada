#pragma once

#include "../core/board.h"
#include "../core/movegen.h"
#include "../tablebase/tb_probe.h"
#include "tt.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace search {

// Score constants
constexpr int SCORE_INFINITE = 32000;
constexpr int SCORE_MATE = 30000;      // Mate at root = 30000, mate in N ply = 30000 - N
constexpr int SCORE_TB_WIN = 29000;    // Tablebase win
constexpr int SCORE_TB_LOSS = -29000;  // Tablebase loss
constexpr int SCORE_SPECIAL = 28000;   // Scores with |score| > this need special TT handling
constexpr int SCORE_DRAW = 0;

// Adjust mate score for ply distance (so we prefer shorter mates)
inline int mate_score(int ply) { return SCORE_MATE - ply; }
inline int mated_score(int ply) { return -SCORE_MATE + ply; }

// Check if score is a "special" score (mate or TB) that needs careful TT handling
inline bool is_special_score(int score) {
  return score > SCORE_SPECIAL || score < -SCORE_SPECIAL;
}

// Check if score indicates a forced mate
inline bool is_mate_score(int score) {
  return score > SCORE_TB_WIN || score < -SCORE_TB_WIN;
}

// Search result
struct SearchResult {
  Move best_move;
  int score;
  int depth;
  std::uint64_t nodes;
  std::uint64_t tb_hits;

  SearchResult() : score(0), depth(0), nodes(0), tb_hits(0) {}
};

// Search statistics
struct SearchStats {
  std::uint64_t nodes = 0;
  std::uint64_t tb_hits = 0;
  std::uint64_t tt_hits = 0;
  std::uint64_t tt_cutoffs = 0;
};

// Evaluation function type
// Takes a board (white to move) and returns a score from white's perspective
using EvalFunc = std::function<int(const Board&)>;

// Random evaluation: reproducible pseudo-random score in [-10000, +10000]
// Used as placeholder until neural network is trained
int random_eval(const Board& board);

// Searcher class - the main search engine
class Searcher {
public:
  // Construct with optional tablebase manager
  // tb_directory: path to directory containing cwdl_*.bin and dtm_*.bin files
  // tb_piece_limit: use WDL tablebases for positions with this many pieces or fewer
  // dtm_piece_limit: use DTM optimal play for positions with this many pieces or fewer
  explicit Searcher(const std::string& tb_directory = "", int tb_piece_limit = 7,
                    int dtm_piece_limit = 6);

  ~Searcher();

  // Set the evaluation function (default is material_eval)
  void set_eval(EvalFunc eval) { eval_ = std::move(eval); }

  // Set transposition table size in MB
  void set_tt_size(std::size_t mb) { tt_ = TranspositionTable(mb); }

  // Clear transposition table
  void clear_tt() { tt_.clear(); }

  // Search to a fixed depth
  SearchResult search(const Board& board, int depth);

  // Search with iterative deepening up to max_depth
  SearchResult search_iterative(const Board& board, int max_depth);

  // Get statistics from last search
  const SearchStats& stats() const { return stats_; }

private:
  // Negamax alpha-beta search
  // Returns score from the perspective of the side to move (white, since board is always flipped)
  // Continues searching beyond depth 0 if captures are available (quiescence)
  int negamax(const Board& board, int depth, int alpha, int beta, int ply);

  // Probe tablebase if available
  // Returns true if position was found, sets score (adjusted for ply)
  bool probe_tb(const Board& board, int ply, int& score);

  // Order moves for better pruning
  void order_moves(std::vector<Move>& moves, const Board& board, const Move& tt_move);

  // Convert DTM to search score
  int dtm_to_score(tablebase::DTM dtm, int ply);

  TranspositionTable tt_;
  EvalFunc eval_;
  SearchStats stats_;

  // Tablebase support
  std::unique_ptr<CompressedTablebaseManager> tb_manager_;
  std::unique_ptr<tablebase::DTMTablebaseManager> dtm_manager_;
  int tb_piece_limit_;
  int dtm_piece_limit_;  // Use DTM optimal play when <= this many pieces
};

} // namespace search
