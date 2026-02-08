#pragma once

#include "../core/board.hpp"
#include "../core/movegen.hpp"
#include "../nn/mlp.hpp"
#include "../tablebase/tb_probe.hpp"
#include "tt.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>

namespace search {

// Search limits
constexpr int MAX_PLY = 128;

// Score constants
constexpr int SCORE_INFINITE = 32000;
constexpr int SCORE_MATE = 30000;      // Mate at root = 30000, mate in N ply = 30000 - N
constexpr int SCORE_TB_WIN = 29000;    // Tablebase win
constexpr int SCORE_TB_LOSS = -29000;  // Tablebase loss
constexpr int SCORE_SPECIAL = 28000;   // Scores with |score| > this need special TT handling

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
  std::vector<Move> pv;  // Principal variation

  SearchResult() : score(0), depth(0), nodes(0), tb_hits(0) {}
};

// Multi-move search result: scores for all root moves at the same depth
struct MultiSearchResult {
  struct MoveScore {
    Move move;
    int score;
  };
  std::vector<MoveScore> moves;  // sorted by score, descending
  int depth = 0;
  std::uint64_t nodes = 0;
  std::uint64_t tb_hits = 0;
};

// Search statistics
struct SearchStats {
  std::uint64_t nodes = 0;
  std::uint64_t tb_hits = 0;
  std::uint64_t tt_hits = 0;
  std::uint64_t tt_cutoffs = 0;

  // Timing (accumulated across all searches)
  double eval_time = 0;      // Time in neural network evaluation
  double tt_probe_time = 0;  // Time probing TT
  double tt_store_time = 0;  // Time storing in TT
  double movegen_time = 0;   // Time generating moves in search
  std::uint64_t eval_calls = 0;
  std::uint64_t tt_probes = 0;
  std::uint64_t tt_stores = 0;
  std::uint64_t movegen_calls = 0;
};

// Evaluation function type
// Takes a board (always stored as white to move) and ply depth
// ply is used to determine the original side: even ply = original white, odd ply = original black
// Returns a score from the perspective of the side to move in the stored board
using EvalFunc = std::function<int(const Board&, int ply)>;

// DTM probe function type for custom tablebase implementations
// Returns DTM value (DTM_UNKNOWN if not found)
using DTMProbeFunc = std::function<tablebase::DTM(const Board&)>;

// Callback for iterative deepening progress updates
// Called after each depth is completed with the current result
using SearchProgressCallback = std::function<void(const SearchResult&)>;

// Exception thrown when search is interrupted
struct SearchInterrupted : std::exception {
  const char* what() const noexcept override { return "search interrupted"; }
};

// Random evaluation: reproducible pseudo-random score in [-10000, +10000]
// Used as placeholder until neural network is trained
int random_eval(const Board& board, int ply);

// Searcher class - the main search engine
class Searcher {
public:
  // Construct with optional DTM tablebase manager and neural network model
  // tb_directory: path to directory containing dtm_*.bin files
  // tb_piece_limit: use DTM tablebases for positions with this many pieces or fewer
  // nn_model_path: path to neural network model (.bin file), empty for random eval
  // dtm_nn_model_path: path to DTM specialist model (.bin file), for 6-7 piece positions
  explicit Searcher(const std::string& tb_directory = "", int tb_piece_limit = 7,
                    const std::string& nn_model_path = "",
                    const std::string& dtm_nn_model_path = "");

  // Construct with pre-loaded external DTM tablebase manager (non-owning, const).
  // The manager must outlive this Searcher and be preloaded before parallel use.
  // Using const pointer guarantees thread-safe read-only access.
  Searcher(const tablebase::DTMTablebaseManager* dtm_tb, int tb_piece_limit,
           const std::string& nn_model_path = "",
           const std::string& dtm_nn_model_path = "");

  ~Searcher();

  // Set the evaluation function (default is material_eval)
  void set_eval(EvalFunc eval) { eval_ = std::move(eval); }

  // Set a custom DTM probe function (for WASM or other custom tablebase implementations)
  // This overrides the built-in DTMTablebaseManager lookup
  void set_dtm_probe(DTMProbeFunc probe, int piece_limit) {
    dtm_probe_func_ = std::move(probe);
    tb_piece_limit_ = piece_limit;
  }

  // Set transposition table size in MB
  void set_tt_size(std::size_t mb) { tt_ = TranspositionTable(mb); }

  // Clear transposition table
  void clear_tt() { tt_.clear(); }

  // Enable verbose output during search
  void set_verbose(bool v) { verbose_ = v; }

  // Set perspective for PV display (true = white's view, false = black's view)
  void set_perspective(bool white) { white_perspective_ = white; }

  // Set the value of a draw from white's perspective
  // Default is 0. Use negative values (e.g., -100) to make the engine avoid draws.
  // Use -10000 to make draws as bad as losses for white.
  void set_draw_score(int score) { draw_score_ = score; }
  int draw_score() const { return draw_score_; }

  // Set whether white is to move at the root of the search (from the game's perspective)
  // This is needed to correctly apply draw_score based on whose turn it originally is
  void set_root_white_to_move(bool white) { root_white_to_move_ = white; }

  // Set external stop flag (for SIGINT handling)
  void set_stop_flag(std::atomic<bool>* flag) { stop_flag_ = flag; }

  // Set callback for iterative deepening progress updates
  void set_progress_callback(SearchProgressCallback cb) { progress_callback_ = std::move(cb); }

  // Check if search should stop and throw if so
  void check_stop() const {
    if (stop_flag_ && stop_flag_->load(std::memory_order_relaxed)) throw SearchInterrupted{};
    if (hard_node_limit_ > 0 && stats_.nodes >= hard_node_limit_) throw SearchInterrupted{};
  }

  // Search with iterative deepening
  // max_depth: maximum search depth (default 100)
  // max_nodes: soft node limit, stops after completing a depth (0 = no limit)
  SearchResult search(const Board& board, int max_depth = 100, std::uint64_t max_nodes = 0);

  // Multi-move search: returns scores for all root moves at the same depth.
  // Uses threshold-based pruning at the root: moves clearly below best - threshold
  // get a fail-low bound instead of an exact score (sufficient for filtering).
  MultiSearchResult search_multi(const Board& board, int max_depth = 100,
                                 std::uint64_t max_nodes = 0, int threshold = 100);

  // Get statistics from last search
  const SearchStats& stats() const { return stats_; }

private:
  // Root search at a fixed depth
  // root_moves is reordered to put the best move first
  SearchResult search_root(const Board& board, MoveList& root_moves, int depth);

  // Root search returning scores for all moves. Uses threshold-based window:
  // exact scores for moves within threshold of best, fail-low bounds for others.
  // Returns the best score. Reorders root_moves/scores with best first.
  int search_root_all(const Board& board, MoveList& root_moves, int depth,
                      int threshold, std::vector<int>& scores);

  // Negamax alpha-beta search
  // Returns score from the perspective of the side to move (white, since board is always flipped)
  // Continues searching beyond depth 0 if captures are available (quiescence)
  int negamax(const Board& board, int depth, int alpha, int beta, int ply);

  // Probe tablebase if available
  // Returns true if position was found, sets score (adjusted for ply)
  bool probe_tb(const Board& board, int ply, int& score);

  // Order moves for better pruning
  void order_moves(MoveList& moves, const Board& board, const Move& tt_move);

  // Extract principal variation from TT
  void extract_pv(const Board& board, std::vector<Move>& pv, int max_depth);

  // Convert DTM to search score
  int dtm_to_score(tablebase::DTM dtm, int ply);

  // Get effective draw score for a given search ply
  // Combines root_white_to_move_ with ply to determine if it's original white's turn
  int effective_draw_score(int ply) const {
    // If root is white's turn and ply is even, or root is black's turn and ply is odd,
    // then it's original white's turn
    bool is_original_white = (root_white_to_move_ == (ply % 2 == 0));
    return is_original_white ? draw_score_ : -draw_score_;
  }

  TranspositionTable tt_;
  EvalFunc eval_;
  SearchStats stats_;

  // DTM tablebase support (owned - for when Searcher loads its own)
  std::unique_ptr<tablebase::DTMTablebaseManager> dtm_manager_owned_;
  // Const pointer for thread-safe read-only access (either points to owned or external)
  const tablebase::DTMTablebaseManager* dtm_manager_ = nullptr;
  // Custom DTM probe function (alternative to dtm_manager_)
  DTMProbeFunc dtm_probe_func_;
  int tb_piece_limit_;  // Use DTM tablebases when <= this many pieces

  // Neural network evaluation
  std::unique_ptr<nn::MLP> nn_model_;       // General evaluation (8+ pieces)
  std::unique_ptr<nn::MLP> dtm_nn_model_;   // DTM specialist (6-7 pieces)

  // Verbose output
  bool verbose_ = false;
  bool white_perspective_ = true;  // For PV display

  // Draw score from white's perspective (default 0)
  int draw_score_ = 0;

  // Whether white is to move at the root (from the game's perspective)
  bool root_white_to_move_ = true;

  // External stop flag (for SIGINT handling)
  std::atomic<bool>* stop_flag_ = nullptr;

  // Progress callback for iterative deepening
  SearchProgressCallback progress_callback_;

  // Hard node limit (0 = no limit)
  std::uint64_t hard_node_limit_ = 0;

  // Killer moves (2 per ply)
  Move killers_[MAX_PLY][2] = {};

  // History heuristic: indexed by [from_sq][to_sq]
  // Lower values are better (moves that cause cutoffs get decremented)
  std::int16_t history_[32][32] = {};

  // Position hash history for repetition detection (indexed by ply)
  // Stores position_hash() (without n_reversible) for each position in the search path
  std::uint64_t pos_hash_history_[MAX_PLY] = {};

};

} // namespace search
