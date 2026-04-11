#pragma once

#include "../core/board.hpp"
#include "../core/movegen.hpp"
#include "../nn/mlp.hpp"
#include "../tablebase/tb_probe.hpp"
#include "tt.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
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
constexpr int SCORE_DRAW = 10000;      // Proven draw range: [-SCORE_DRAW, +SCORE_DRAW]

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

// Simple material eval for proven draws with Beal-effect noise.
// Deterministic: noise is derived from position hash so TT entries stay consistent.
inline int draw_eval(const Board& board) {
  int wp = std::popcount(board.whitePawns());
  int wk = std::popcount(board.whiteQueens());
  int bp = std::popcount(board.blackPawns());
  int bk = std::popcount(board.blackQueens());
  int material = (wp - bp) * 100 + (wk - bk) * 320;
  int noise = static_cast<int>(board.hash() % 21) - 10;
  return material + noise;
}

// Leaf eval inside a known-win subtree (side to move wins per WDL TB).
// Weights are asymmetric — losing-side pieces are worth more than winning-side
// pieces, so the engine strictly prefers capturing opponent material over
// keeping its own (i.e., prefers simplification).
//
// The -500 offset on the base exists to keep the maximum strictly below
// SCORE_TB_WIN. With max material adjustment +372 (12 winner queens vs 0
// opponent pieces) the peak value is 28872 - ply, which satisfies
// is_mate_score() == false. Without the offset the peak would reach
// 29372 - ply > SCORE_TB_WIN, which would trip the iterative-deepening
// "found a mate, stop here" break in Searcher::search and prevent the
// depth-reduced in-tree search from looking further past the conversion
// boundary.
//
// Range: 28500 - ply + [-552, +372]. The upper bound stays below
// SCORE_TB_WIN for any ply. The lower bound dips below SCORE_SPECIAL only
// for degenerate material imbalances (winner heavily out-materialled by
// the loser) that don't correspond to real WDL=WIN positions, so
// is_special_score() effectively remains true in practice.
inline int tb_win_score(const Board& board, int ply) {
  int wp = std::popcount(board.whitePawns());
  int wq = std::popcount(board.whiteQueens());
  int bp = std::popcount(board.blackPawns());
  int bq = std::popcount(board.blackQueens());
  return SCORE_TB_WIN - 500 - ply + 10 * wp + 31 * wq - 15 * bp - 46 * bq;
}

// Leaf eval inside a known-loss subtree (side to move loses per WDL TB).
// Exact sign-flipped mirror of tb_win_score (see notes there), expressed
// from the side-to-move's viewpoint.
inline int tb_loss_score(const Board& board, int ply) {
  int wp = std::popcount(board.whitePawns());
  int wq = std::popcount(board.whiteQueens());
  int bp = std::popcount(board.blackPawns());
  int bq = std::popcount(board.blackQueens());
  return -SCORE_TB_WIN + 500 + ply - 10 * bp - 31 * bq + 15 * wp + 46 * wq;
}

// Check if score is in the proven-draw range
inline bool is_proven_draw(int score) {
  return score >= -SCORE_DRAW && score <= SCORE_DRAW;
}

// Shift NN eval score (in [-10000, +10000]) to the undecided range
// Positive scores go to [+10001, +20000], negative to [-20000, -10001]
// Zero stays at 0 (in the draw range — ambiguous but rare and harmless)
inline int to_undecided(int nn_score) {
  if (nn_score > 0) return nn_score + SCORE_DRAW;
  if (nn_score < 0) return nn_score - SCORE_DRAW;
  return 0;
}

// Strip the undecided offset for display purposes
inline int undecided_to_display(int score) {
  if (score > SCORE_DRAW) return score - SCORE_DRAW;
  if (score < -SCORE_DRAW) return score + SCORE_DRAW;
  return score;
}

// Collapse search score to the [-10000, +10000] NN-eval scale:
//   Proven draws → 0
//   Undecided    → strip ±SCORE_DRAW offset (back to NN eval)
//   TB win/mate  → +10000
//   TB loss/mated→ -10000
// Useful for book Q values, eval bars, and any context that needs a
// single unified scale without the proven-draw / undecided distinction.
inline int score_to_normalized(int score) {
  if (is_proven_draw(score)) return 0;
  if (is_special_score(score)) return (score > 0) ? SCORE_DRAW : -SCORE_DRAW;
  // Undecided range: strip offset
  return (score > 0) ? score - SCORE_DRAW : score + SCORE_DRAW;
}

// Tablebase verdict carried down the search recursion.
// Set by probe_wdl when the TB resolves a position; propagated to descendants
// (negated for the side-to-move flip). Drives verdict-aware leaf eval,
// depth-reduction inside known subtrees, and TT gating.
enum class KnownVerdict { UNKNOWN, WIN, LOSS, DRAW };

// Flip a verdict for the opposite side to move (negamax recursion).
// WIN <-> LOSS; DRAW and UNKNOWN stay the same.
constexpr KnownVerdict negate(KnownVerdict v) {
  switch (v) {
    case KnownVerdict::WIN:  return KnownVerdict::LOSS;
    case KnownVerdict::LOSS: return KnownVerdict::WIN;
    default:                 return v;
  }
}

// Secondary search phase indicator
enum class SearchPhase { PRIMARY, SECONDARY_WINNING, SECONDARY_LOSING };

// Search result
struct SearchResult {
  Move best_move;
  int score;
  int depth;
  std::uint64_t nodes;
  std::uint64_t tb_hits;
  std::vector<Move> pv;  // Principal variation
  std::vector<std::pair<Move, int>> root_scores;  // All root move scores (ponder mode)
  SearchPhase phase = SearchPhase::PRIMARY;

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

// WDL probe function type for compressed WDL tablebases
// Returns: 1=WIN, 0=DRAW, -1=LOSS, or nullopt if not found
using WDLProbeFunc = std::function<std::optional<int>(const Board&)>;

// Callback for iterative deepening progress updates
// Called after each depth is completed with the current result
using SearchProgressCallback = std::function<void(const SearchResult&)>;

// Exception thrown when search is interrupted
struct SearchInterrupted : std::exception {
  const char* what() const noexcept override { return "search interrupted"; }
};

// Time and node control for search limits
struct TimeControl {
  // Soft limits: stop after completing a depth if exceeded
  std::uint64_t soft_node_limit = 0;  // 0 = no limit
  double soft_time_seconds = 0;       // 0 = no limit

  // Hard limits: throw SearchInterrupted mid-search
  std::uint64_t hard_node_limit = 0;  // 0 = no limit
  double hard_time_seconds = 0;       // 0 = no limit

  // Node count at which to next check time/hard limits
  std::uint64_t node_count_for_next_check = 0;

  // Record start time and set first check point
  void start();

  // Called during search when nodes >= node_count_for_next_check
  // May throw SearchInterrupted, updates next check point
  void check(std::uint64_t nodes);

  // True if any soft limit has been exceeded
  bool exceeded_soft(std::uint64_t nodes) const;

  // Factory: node-based limits (hard = soft * 5 if not specified)
  static TimeControl with_nodes(std::uint64_t soft, std::uint64_t hard = 0);

  // Factory: time-based limits
  static TimeControl with_time(double soft_seconds, double hard_seconds = 0);

  // Elapsed time since start() was called
  double elapsed_seconds() const;

private:
  std::chrono::steady_clock::time_point start_time_;
  static constexpr std::uint64_t CHECK_INTERVAL = 4096;
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
  explicit Searcher(const std::string& tb_directory = "", int tb_piece_limit = 7,
                    const std::string& nn_model_path = "");

  // Construct with pre-loaded external DTM tablebase manager (non-owning, const).
  // The manager must outlive this Searcher and be preloaded before parallel use.
  // Using const pointer guarantees thread-safe read-only access.
  Searcher(const tablebase::DTMTablebaseManager* dtm_tb, int tb_piece_limit,
           const std::string& nn_model_path = "");

  ~Searcher();

  // Set the evaluation function (default is material_eval)
  void set_eval(EvalFunc eval) { eval_ = std::move(eval); }

  // Set a custom DTM probe function (for WASM or other custom tablebase implementations)
  // This overrides the built-in DTMTablebaseManager lookup
  void set_dtm_probe(DTMProbeFunc probe, int piece_limit) {
    dtm_probe_func_ = std::move(probe);
    tb_piece_limit_ = piece_limit;
  }

  // Set a WDL probe function for compressed WDL tablebases (6-7 piece endgames)
  void set_wdl_probe(WDLProbeFunc probe, int piece_limit) {
    wdl_probe_func_ = std::move(probe);
    wdl_piece_limit_ = piece_limit;
  }

  // Set transposition table size in MB
  void set_tt_size(std::size_t mb) { tt_ = TranspositionTable(mb); }

  // Clear transposition table
  void clear_tt() { tt_.clear(); }

  // Enable verbose output during search
  void set_verbose(bool v) { verbose_ = v; }

  // Set perspective for PV display (true = white's view, false = black's view)
  void set_perspective(bool white) { white_perspective_ = white; }


  // Set external stop flag (for SIGINT handling)
  void set_stop_flag(std::atomic<bool>* flag) { stop_flag_ = flag; }

  // Set callback for iterative deepening progress updates
  void set_progress_callback(SearchProgressCallback cb) { progress_callback_ = std::move(cb); }

  // Set analysis mode: search even when there's only one legal move
  void set_analyze_mode(bool v) { analyze_mode_ = v; }

  // Set ponder mode: search each root move with full window to populate TT for all lines
  void set_ponder_mode(bool v) { ponder_mode_ = v; }

  // Set draw value for Espada/Broquel proving
  // Positive = draws are wins for root side; negative = draws are losses
  void set_draw_value(int v) { draw_value_ = v; }

  // Check if search should stop and throw if so
  void check_stop() {
    if (stop_flag_ && stop_flag_->load(std::memory_order_relaxed)) throw SearchInterrupted{};
    if (stats_.nodes >= tc_.node_count_for_next_check) {
      tc_.check(stats_.nodes);
    }
  }

  // Search with iterative deepening
  // max_depth: maximum search depth (default 100)
  // tc: time and node control (default unlimited)
  SearchResult search(const Board& board, int max_depth = 100,
                      const TimeControl& tc = TimeControl{});

  // Multi-move search: returns scores for all root moves at the same depth.
  // Uses threshold-based pruning at the root: moves clearly below best - threshold
  // get a fail-low bound instead of an exact score (sufficient for filtering).
  MultiSearchResult search_multi(const Board& board, int max_depth = 100,
                                 const TimeControl& tc = TimeControl{}, int threshold = 100);

  // Get statistics from last search
  const SearchStats& stats() const { return stats_; }

private:
  // Secondary search for WDL tablebase positions
  // After a primary search returns a WDL score, re-search with WDL disabled
  // so the NN guides move selection among theoretically equivalent moves
  SearchResult secondary_search(const Board& board, MoveList& root_moves,
                                const SearchResult& primary,
                                int max_depth);

  // Root search at a fixed depth
  // root_moves is reordered to put the best move first
  SearchResult search_root(const Board& board, MoveList& root_moves, int depth);

  // Root search returning scores for all moves. Uses threshold-based window:
  // exact scores for moves within threshold of best, fail-low bounds for others.
  // Returns the best score. Reorders root_moves/scores with best first.
  int search_root_all(const Board& board, MoveList& root_moves, int depth,
                      int threshold, std::vector<int>& scores,
                      bool full_window = false);

  // Negamax alpha-beta search
  // Returns score from the perspective of the side to move (white, since board is always flipped)
  // Continues searching beyond depth 0 if captures are available (quiescence)
  // `verdict` carries the tablebase verdict for this subtree (from STM's perspective),
  // set by the parent's probe_wdl. When non-UNKNOWN the search reduces depth,
  // skips the WDL probe, skips the TT, and uses a verdict-specific leaf eval.
  int negamax(const Board& board, int depth, int alpha, int beta, int ply,
              KnownVerdict verdict);

  // Probe DTM tablebase if available
  // Returns true if position was found, sets score (adjusted for ply)
  bool probe_tb(const Board& board, int ply, int& score);

  // WDL probe result.
  //   NOT_FOUND    — no TB information for this position
  //   SCORE_READY  — score is finalized (shallow draws, Espada/Broquel draws,
  //                  window-cutoff draws); caller should return it immediately
  //   WIN_VERDICT  — STM wins per WDL TB; caller enters known-win subtree
  //   LOSS_VERDICT — STM loses per WDL TB; caller enters known-loss subtree
  //   DRAW_VERDICT — proven draw, deep enough to keep searching; caller enters
  //                  known-draw subtree
  enum class WDLProbeResult {
    NOT_FOUND,
    SCORE_READY,
    WIN_VERDICT,
    LOSS_VERDICT,
    DRAW_VERDICT
  };

  // Probe WDL tablebase if available (for ≤7 piece endgames).
  // `score` is written only on SCORE_READY; on all other results it is left
  // untouched and the caller determines the value from the verdict itself.
  WDLProbeResult probe_wdl(const Board& board, int ply, int depth,
                            int alpha, int beta, int& score);

  // Order moves for better pruning
  void order_moves(MoveList& moves, const Board& board, const Move& tt_move);

  // Extract principal variation from TT
  void extract_pv(const Board& board, std::vector<Move>& pv, int max_depth);

  // Convert DTM to search score
  int dtm_to_score(tablebase::DTM dtm, int ply);


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

  // WDL probe function for compressed WDL tablebases
  WDLProbeFunc wdl_probe_func_;
  int wdl_piece_limit_ = 0;  // Use WDL tablebases when <= this many pieces

  // Neural network evaluation
  std::unique_ptr<nn::MLP> nn_model_;

  // Verbose output
  bool verbose_ = false;
  bool white_perspective_ = true;  // For PV display


  // Analysis mode: search even with only one legal move
  bool analyze_mode_ = false;

  // Ponder mode: search each root move with full window
  bool ponder_mode_ = false;

  // Draw value for Espada/Broquel mode
  // 0 = normal draws; nonzero = draws are decisive (from root's perspective)
  int draw_value_ = 0;

  // External stop flag (for SIGINT handling)
  std::atomic<bool>* stop_flag_ = nullptr;

  // Progress callback for iterative deepening
  SearchProgressCallback progress_callback_;

  // Active time control for current search
  TimeControl tc_;

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
