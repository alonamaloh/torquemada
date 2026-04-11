#include "search.hpp"
#include <algorithm>
#include <bit>
#include <cstdio>
#include <cstring>
#include <iostream>

namespace search {

// --- TimeControl implementation ---

void TimeControl::start() {
  start_time_ = std::chrono::steady_clock::now();
  node_count_for_next_check = (hard_node_limit > 0)
    ? std::min(hard_node_limit, CHECK_INTERVAL)
    : (hard_time_seconds > 0 ? CHECK_INTERVAL : UINT64_MAX);
}

void TimeControl::check(std::uint64_t nodes) {
  // Check hard node limit
  if (hard_node_limit > 0 && nodes >= hard_node_limit) {
    throw SearchInterrupted{};
  }
  // Check hard time limit
  if (hard_time_seconds > 0) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed >= hard_time_seconds) {
      throw SearchInterrupted{};
    }
  }
  // Schedule next check
  node_count_for_next_check = nodes + CHECK_INTERVAL;
  if (hard_node_limit > 0 && node_count_for_next_check > hard_node_limit) {
    node_count_for_next_check = hard_node_limit;
  }
}

bool TimeControl::exceeded_soft(std::uint64_t nodes) const {
  if (soft_node_limit > 0 && nodes >= soft_node_limit) return true;
  if (soft_time_seconds > 0) {
    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - start_time_).count();
    if (elapsed >= soft_time_seconds) return true;
  }
  return false;
}

TimeControl TimeControl::with_nodes(std::uint64_t soft, std::uint64_t hard) {
  TimeControl tc;
  tc.soft_node_limit = soft;
  tc.hard_node_limit = (hard > 0) ? hard : (soft > 0 ? soft * 5 : 0);
  return tc;
}

double TimeControl::elapsed_seconds() const {
  auto now = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(now - start_time_).count();
}

TimeControl TimeControl::with_time(double soft_seconds, double hard_seconds) {
  TimeControl tc;
  tc.soft_time_seconds = soft_seconds;
  tc.hard_time_seconds = hard_seconds;
  return tc;
}

// Random evaluation: reproducible pseudo-random score derived from position hash
// Returns a score in the range [-10000, +10000] based on the hash
int random_eval(const Board& board, int /*ply*/) {
  std::uint64_t h = board.hash() + 1;

  // Mix the hash to get good distribution
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;

  // Convert to signed score in range [-10000, +10000]
  std::int64_t raw = static_cast<std::int64_t>(h);
  return static_cast<int>(raw % 10001);
}

Searcher::Searcher(const std::string& tb_directory, int tb_piece_limit,
                   const std::string& nn_model_path)
    : tt_(64), eval_(random_eval), tb_piece_limit_(tb_piece_limit) {
  if (!tb_directory.empty() && tb_piece_limit > 0) {
    dtm_manager_owned_ = std::make_unique<tablebase::DTMTablebaseManager>(tb_directory);
    dtm_manager_owned_->preload(tb_piece_limit);
    dtm_manager_ = dtm_manager_owned_.get();
  }

  if (!nn_model_path.empty()) {
    nn_model_ = std::make_unique<nn::MLP>(nn_model_path);
    eval_ = [this](const Board& board, int /*ply*/) {
      return nn_model_->evaluate(board);
    };
  }
}

Searcher::Searcher(const tablebase::DTMTablebaseManager* dtm_tb, int tb_piece_limit,
                   const std::string& nn_model_path)
    : tt_(64), eval_(random_eval), dtm_manager_(dtm_tb), tb_piece_limit_(tb_piece_limit) {
  if (!nn_model_path.empty()) {
    nn_model_ = std::make_unique<nn::MLP>(nn_model_path);
    eval_ = [this](const Board& board, int /*ply*/) {
      return nn_model_->evaluate(board);
    };
  }
}

Searcher::~Searcher() = default;

int Searcher::dtm_to_score(tablebase::DTM dtm, int ply) {
  if (dtm == tablebase::DTM_UNKNOWN) {
    return 0;  // Shouldn't happen
  }
  if (dtm == tablebase::DTM_DRAW) {
    return 0;
  }
  if (dtm == tablebase::DTM_LOSS_TERMINAL) {
    // Terminal loss: no legal moves, lost right now
    return -SCORE_MATE + ply;
  }
  if (dtm > 0) {
    // Win in dtm moves - convert to mate-like score
    // DTM is in "moves" (winning side's moves), score uses plies from root
    return SCORE_MATE - ply - (2 * dtm - 1);
  } else {
    // Loss in -dtm moves
    return -SCORE_MATE + ply + (2 * (-dtm));
  }
}

bool Searcher::probe_tb(const Board& board, int ply, int& score) {
  // Check if we have any DTM source available
  if (!dtm_manager_ && !dtm_probe_func_) return false;

  int piece_count = std::popcount(board.allPieces());
  if (piece_count > tb_piece_limit_) return false;

  // Try custom probe function first (for WASM), then fall back to manager
  tablebase::DTM dtm;
  if (dtm_probe_func_) {
    dtm = dtm_probe_func_(board);
  } else {
    dtm = dtm_manager_->lookup_dtm(board);
  }
  if (dtm == tablebase::DTM_UNKNOWN) return false;

  if (dtm == tablebase::DTM_DRAW) {
    // Don't short-circuit draws — let probe_wdl handle them (returning
    // SCORE_READY with a heuristic draw score, or DRAW_VERDICT for a
    // depth-reduced known-draw search) so move ordering stays useful.
    return false;
  }

  stats_.tb_hits++;
  score = dtm_to_score(dtm, ply);
  return true;
}

Searcher::WDLProbeResult Searcher::probe_wdl(const Board& board, int ply, int depth,
                                              int alpha, int beta, int& score) {
  if (!wdl_probe_func_) return WDLProbeResult::NOT_FOUND;

  int piece_count = std::popcount(board.allPieces());
  if (piece_count > wdl_piece_limit_) return WDLProbeResult::NOT_FOUND;

  // DTM wins/losses are already handled by probe_tb (called before probe_wdl).
  // DTM draws return false from probe_tb so they fall through here for
  // draw-verdict handling, same as WDL draws.
  auto result = wdl_probe_func_(board);
  if (!result.has_value()) return WDLProbeResult::NOT_FOUND;

  int wdl = *result;

  if (wdl == 0) {
    // Proven draw.
    stats_.tb_hits++;

    // Espada/Broquel mode: draws are decisive
    if (draw_value_ != 0) {
      score = (ply % 2 == 0) ? draw_value_ : -draw_value_;
      return WDLProbeResult::SCORE_READY;
    }

    // Window cutoff: if the search needs something better than any draw,
    // or any draw causes a cutoff, return immediately.
    if (alpha > SCORE_DRAW || beta < -SCORE_DRAW) {
      score = 0;
      return WDLProbeResult::SCORE_READY;
    }

    // Shallow depth: use simple material eval for draw ordering.
    if (depth <= 3) {
      score = draw_eval(board);
      return WDLProbeResult::SCORE_READY;
    }

    // Deep: enter known-draw subtree and keep searching with reduced depth.
    return WDLProbeResult::DRAW_VERDICT;
  }

  // WIN/LOSS: always trust when the TB has data. The caller (negamax) handles
  // the "no progress yet" case by tagging the verdict as WIN_BEFORE/LOSS_BEFORE
  // rather than WIN_PAST/LOSS_PAST — the 200-point BEFORE→PAST gap at the leaf
  // biases the search toward finding an irreversible move rather than stalling.
  // The underlying WDL tablebase doesn't model the 50-move rule, but the
  // BEFORE/PAST gradient gives the engine a concrete incentive to convert.
  stats_.tb_hits++;
  return (wdl > 0) ? WDLProbeResult::WIN_VERDICT : WDLProbeResult::LOSS_VERDICT;
}

int Searcher::negamax(const Board& board, int depth, int alpha, int beta, int ply,
                      KnownVerdict verdict) {
  check_stop();  // Throws SearchInterrupted if we should stop

  stats_.nodes++;

  // Store position hash for repetition detection (hash without n_reversible)
  std::uint64_t pos_hash = board.position_hash();
  if (ply < MAX_PLY) {
    pos_hash_history_[ply] = pos_hash;
  }

  // Check for repetition: scan back through the search path
  // Start 4 plies back (minimum for a repetition), step by 2, up to n_reversible plies
  // Note: we can only have a repetition if no irreversible move was made
  if (board.n_reversible >= 4 && ply >= 4) {
    int max_back = std::min(static_cast<int>(board.n_reversible), ply);
    for (int back = 4; back <= max_back; back += 2) {
      if (pos_hash_history_[ply - back] == pos_hash) {
        // Found a repetition - return draw score (or draw_value_ for Espada/Broquel)
        return (ply % 2 == 0) ? draw_value_ : -draw_value_;
      }
    }
  }

  // If we inherited a *_BEFORE verdict and the parent just played an
  // irreversible move (n_reversible == 0 at this child), re-probe the WDL TB
  // once to confirm the verdict for the new material configuration. The
  // re-probe catches blundering captures — if our old verdict says LOSS_BEFORE
  // but the irreversible move turned it into a WIN (or vice versa), the new
  // probe returns the correct side-to-move verdict and we transition to PAST.
  // Subsequent irreversible moves inside a *_PAST subtree are NOT re-probed.
  int tb_score;
  if ((verdict == KnownVerdict::WIN_BEFORE || verdict == KnownVerdict::LOSS_BEFORE)
      && board.n_reversible == 0) {
    auto wdl = probe_wdl(board, ply, depth, alpha, beta, tb_score);
    switch (wdl) {
      case WDLProbeResult::NOT_FOUND:
        verdict = KnownVerdict::UNKNOWN;  // defensive; shouldn't happen in practice
        break;
      case WDLProbeResult::SCORE_READY:
        return tb_score;
      case WDLProbeResult::WIN_VERDICT:
        verdict = KnownVerdict::WIN_PAST;
        break;
      case WDLProbeResult::LOSS_VERDICT:
        verdict = KnownVerdict::LOSS_PAST;
        break;
      case WDLProbeResult::DRAW_VERDICT:
        verdict = KnownVerdict::DRAW;
        break;
    }
  }

  // Inside a known-verdict subtree (inherited from parent, possibly refreshed
  // above), reduce depth by 2 before doing any further work. Combined with the
  // natural `depth - 1` on recursion, this makes each known-endgame ply consume
  // 3 effective plies of depth — the search peeks a few moves past the
  // conversion boundary without spending as much effort as on undecided
  // branches.
  if (verdict != KnownVerdict::UNKNOWN && depth > 0) {
    depth -= 2;
  }

  // Check for tablebase hit (before TT to get exact values).
  // DTM first (exact distance-to-mate), then WDL (game-theoretic value).
  if (probe_tb(board, ply, tb_score)) {
    return tb_score;
  }

  // WDL probe — only when verdict is still UNKNOWN. Once inside a known-verdict
  // subtree we already have the answer; skipping the probe saves the lookup.
  // On first entry into a known subtree we apply the same depth -= 2 reduction
  // that inherited subtrees get at the top of negamax. The new verdict is the
  // PAST flavor if the move that got us here was irreversible (n_reversible
  // == 0), otherwise the BEFORE flavor.
  if (verdict == KnownVerdict::UNKNOWN) {
    auto wdl = probe_wdl(board, ply, depth, alpha, beta, tb_score);
    switch (wdl) {
      case WDLProbeResult::NOT_FOUND:
        break;
      case WDLProbeResult::SCORE_READY:
        return tb_score;
      case WDLProbeResult::WIN_VERDICT:
        verdict = (board.n_reversible == 0) ? KnownVerdict::WIN_PAST
                                             : KnownVerdict::WIN_BEFORE;
        if (depth > 0) depth -= 2;
        break;
      case WDLProbeResult::LOSS_VERDICT:
        verdict = (board.n_reversible == 0) ? KnownVerdict::LOSS_PAST
                                             : KnownVerdict::LOSS_BEFORE;
        if (depth > 0) depth -= 2;
        break;
      case WDLProbeResult::DRAW_VERDICT:
        verdict = KnownVerdict::DRAW;
        if (depth > 0) depth -= 2;
        break;
    }
  }

  // Save original alpha for correct TT flag computation
  const int original_alpha = alpha;

  // Probe transposition table — only in UNKNOWN positions. The key is
  // `pos_hash`, which is the position hash without n_reversible. Inside a
  // known-verdict subtree we don't touch the TT at all: the leaf values are
  // material-weighted and ply-adjusted, and the search depth is reduced per
  // ply by the known-verdict clock, so a cached score would not generalize
  // across different paths to the same position.
  std::uint64_t key = pos_hash;
  TTEntry tt_entry;
  CompactMove tt_compact_move = 0;

  if (verdict == KnownVerdict::UNKNOWN && tt_.probe(key, tt_entry)) {
    stats_.tt_hits++;
    tt_compact_move = tt_entry.best_move;  // Always use best move for ordering

    // Only use TT scores when n_reversible == 0 to avoid search instability
    // from history-dependent effects (repetition detection, score scaling)
    if (board.n_reversible == 0 && tt_entry.depth >= depth) {
      int tt_score = tt_entry.score;

      switch (tt_entry.flag) {
        case TTFlag::EXACT:
          stats_.tt_cutoffs++;
          return tt_score;
        case TTFlag::LOWER_BOUND:
          if (tt_score >= beta) {
            stats_.tt_cutoffs++;
            return tt_score;
          }
          alpha = std::max(alpha, tt_score);
          break;
        case TTFlag::UPPER_BOUND:
          if (tt_score <= alpha) {
            stats_.tt_cutoffs++;
            return tt_score;
          }
          beta = std::min(beta, tt_score);
          break;
        default:
          break;
      }
    }
  }
  
  // Generate moves
  MoveList moves;
  generateMoves(board, moves);
  
  // Terminal node - no moves means loss
  if (moves.empty()) {
    return mated_score(ply);
  }

  // Leaf node: only evaluate when depth <= 0 AND no captures available.
  // If captures exist, continue searching (quiescence).
  bool has_captures = moves[0].isCapture();
  if (depth <= 0 && !has_captures) {
    switch (verdict) {
      case KnownVerdict::WIN_PAST:
        // We've confirmed a winning irreversible move on this branch.
        return tb_win_score(board, ply);
      case KnownVerdict::WIN_BEFORE:
        // WDL says we win but the path from the root has been all reversible.
        // Bias 200 points toward zero so branches that reach WIN_PAST (and
        // therefore demonstrate concrete progress) are strictly preferred.
        return tb_win_score(board, ply) - 200;
      case KnownVerdict::LOSS_PAST:
        return tb_loss_score(board, ply);
      case KnownVerdict::LOSS_BEFORE:
        // Symmetric: bias 200 toward zero. From the losing side's own
        // perspective this makes BEFORE less bad than PAST (stalling is
        // preferable to marching toward a confirmed loss), and negamax
        // propagation turns that into the winning side preferring PAST.
        return tb_loss_score(board, ply) + 200;
      case KnownVerdict::DRAW: {
        // Inside a proven draw the game-theoretic value is 0, but we still
        // want the engine to play the most practically difficult line (to
        // give a blundering opponent the best chance to go wrong). The raw
        // NN eval provides that ordering; we clamp it to the proven-draw
        // range so no value leaks out of the draw subtree.
        int score = eval_(board, ply);
        return std::max(-SCORE_DRAW, std::min(SCORE_DRAW, score));
      }
      case KnownVerdict::UNKNOWN: {
        int score = eval_(board, ply);
        // Scale score towards draw based on n_reversible to encourage progress.
        // The more reversible moves without progress, the smaller the score.
        // Scale factor: (256 - n_reversible) / 256, capped at 50 reversible moves.
        if (board.n_reversible > 0 && !is_special_score(score)) {
          int scale = std::max(256 - static_cast<int>(board.n_reversible) * 4, 56);  // Min 56/256 ~= 22%
          score = score * scale / 256;
        }
        return to_undecided(score);
      }
    }
  }

  // Move ordering: TT move first, then killers
  auto insert_pos = moves.begin();

  // Bring TT move to the front
  if (tt_compact_move != 0) {
    for (auto it = insert_pos; it != moves.end(); ++it) {
      if (compact_matches(tt_compact_move, *it)) {
        std::swap(*insert_pos, *it);
        ++insert_pos;
        break;
      }
    }
  }

  // Bring killer moves next (only for non-captures, and if not already TT move)
  if (ply < MAX_PLY) {
    for (int k = 0; k < 2; ++k) {
      const Move& killer = killers_[ply][k];
      if (killer.from_xor_to == 0) continue;
      for (auto it = insert_pos; it != moves.end(); ++it) {
        if (it->from_xor_to == killer.from_xor_to && !it->isCapture()) {
          std::swap(*insert_pos, *it);
          ++insert_pos;
          break;
        }
      }
    }
  }

  // Sort remaining moves by history (ascending - lower/negative values first)
  // Note: from_xor_to can be 0 for circular captures, skip those in sorting
  std::sort(insert_pos, moves.end(), [this](const Move& a, const Move& b) {
    if (a.from_xor_to == 0 || b.from_xor_to == 0) return false;
    int a_sq1 = __builtin_ctz(a.from_xor_to);
    int a_sq2 = 31 - __builtin_clz(a.from_xor_to);
    int b_sq1 = __builtin_ctz(b.from_xor_to);
    int b_sq2 = 31 - __builtin_clz(b.from_xor_to);
    return history_[a_sq1][a_sq2] < history_[b_sq1][b_sq2];
  });

  int best_score = -SCORE_INFINITE;
  Move best_move;
  Move first_move = moves[0];
  bool is_first = true;

  const KnownVerdict child_verdict = negate(verdict);
  for (const Move& move : moves) {
    Board child = makeMove(board, move);
    int score;

    if (is_first) {
      // Search first move with full window
      score = -negamax(child, depth - 1, -beta, -alpha, ply + 1, child_verdict);
    } else {
      // LMR: reduce depth for late non-capture moves when depth is sufficient
      int reduction = (depth >= 3 && !move.isCapture()) ? 1 : 0;
      // PVS: try null-window search first, possibly at reduced depth
      score = -negamax(child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, child_verdict);
      // If it fails high, re-search with full window and full depth
      if (score > alpha) {
        score = -negamax(child, depth - 1, -beta, -alpha, ply + 1, child_verdict);
      }
    }

    if (score > best_score) {
      best_score = score;
      best_move = move;

      if (score > alpha) {
        alpha = score;

        if (alpha >= beta) {
          // Store killer move (only for quiet moves)
          if (!move.isCapture() && ply < MAX_PLY) {
            if (killers_[ply][0].from_xor_to != move.from_xor_to) {
              killers_[ply][1] = killers_[ply][0];
              killers_[ply][0] = move;
            }
          }
          // Update history (only if not first move, and skip circular captures)
          if (!is_first && first_move.from_xor_to != 0 && move.from_xor_to != 0) {
            int f1_sq1 = __builtin_ctz(first_move.from_xor_to);
            int f1_sq2 = 31 - __builtin_clz(first_move.from_xor_to);
            int c_sq1 = __builtin_ctz(move.from_xor_to);
            int c_sq2 = 31 - __builtin_clz(move.from_xor_to);
            // Increment first move (should have been tried later)
            if (history_[f1_sq1][f1_sq2] < INT16_MAX) {
              history_[f1_sq1][f1_sq2]++;
            }
            // Decrement cutoff move (should have been tried earlier)
            if (history_[c_sq1][c_sq2] > INT16_MIN) {
              history_[c_sq1][c_sq2]--;
            }
          }
          break;  // Beta cutoff
        }
      }
    }
    is_first = false;
  }

  // Store in transposition table — only in UNKNOWN positions. Known-verdict
  // subtrees don't use the TT (their leaf scores are material-weighted and
  // path-dependent via the depth-reduction clock, so a cached value would not
  // generalize across paths).
  if (verdict == KnownVerdict::UNKNOWN) {
    // Compute TT flag from original alpha (before TT probe may have raised it)
    TTFlag flag = best_score <= original_alpha ? TTFlag::UPPER_BOUND
                : best_score >= beta           ? TTFlag::LOWER_BOUND
                                               : TTFlag::EXACT;

    // When n_reversible > 0, store with a trivial bound (>= -INF) so the best move
    // is preserved for ordering, but the score won't cause any cutoffs. This
    // avoids search instability from history-dependent effects.
    if (board.n_reversible == 0) {
      // For special scores (mate propagating up from terminal mated_score or
      // from a child's known subtree), only store bounds, not exact values,
      // because these scores are relative to the root and may not be valid
      // when accessed from a different search path.
      TTFlag store_flag = flag;
      if (is_special_score(best_score) && flag == TTFlag::EXACT) {
        store_flag = (best_score > 0) ? TTFlag::LOWER_BOUND : TTFlag::UPPER_BOUND;
      }
      tt_.store(key, best_score, depth, store_flag, best_move);
    } else {
      // Store with trivial lower bound - preserves best move for ordering
      tt_.store(key, -SCORE_INFINITE, depth, TTFlag::LOWER_BOUND, best_move);
    }
  }

  return best_score;
}

void Searcher::extract_pv(const Board& board, std::vector<Move>& pv, int max_depth) {
  pv.clear();
  Board pos = board;
  std::uint64_t seen_keys[64];  // To detect cycles
  int seen_count = 0;

  for (int i = 0; i < max_depth && seen_count < 64; ++i) {
    std::uint64_t key = pos.position_hash();  // Use position_hash to match TT key

    // Check for cycle
    for (int j = 0; j < seen_count; ++j) {
      if (seen_keys[j] == key) return;
    }
    seen_keys[seen_count++] = key;

    TTEntry entry;
    if (!tt_.probe(key, entry) || entry.best_move == 0) {
      return;
    }

    // Find the matching move from legal moves
    MoveList moves;
    generateMoves(pos, moves);
    Move* found_move = nullptr;
    for (Move& m : moves) {
      if (compact_matches(entry.best_move, m)) {
        found_move = &m;
        break;
      }
    }
    if (!found_move) return;

    pv.push_back(*found_move);
    pos = makeMove(pos, *found_move);
  }
}

SearchResult Searcher::search_root(const Board& board, MoveList& moves, int depth) {
  SearchResult result;
  result.depth = depth;

  // Store root position hash for repetition detection
  pos_hash_history_[0] = board.position_hash();

  int alpha = -SCORE_INFINITE;
  int beta = SCORE_INFINITE;

  for (std::size_t i = 0; i < moves.size(); ++i) {
    const Move& move = moves[i];
    Board child = makeMove(board, move);
    int score = -negamax(child, depth - 1, -beta, -alpha, 1, KnownVerdict::UNKNOWN);

    if (score > alpha) {
      alpha = score;
      result.best_move = move;
      result.score = score;

      // Rotate this move to the front for better ordering in next iteration
      if (i > 0) {
        std::rotate(moves.begin(), moves.begin() + i, moves.begin() + i + 1);
      }
    }
  }

  result.nodes = stats_.nodes;
  result.tb_hits = stats_.tb_hits;

  // Extract PV from TT (root move + continuation from TT)
  if (result.best_move.from_xor_to != 0) {
    result.pv.push_back(result.best_move);
    Board child = makeMove(board, result.best_move);
    std::vector<Move> continuation;
    extract_pv(child, continuation, depth - 1);
    result.pv.insert(result.pv.end(), continuation.begin(), continuation.end());
  }

  return result;
}

int Searcher::search_root_all(const Board& board, MoveList& root_moves, int depth,
                              int threshold, std::vector<int>& scores,
                              bool full_window) {
  pos_hash_history_[0] = board.position_hash();
  scores.resize(root_moves.size());

  int best_score = -SCORE_INFINITE;
  std::size_t best_idx = 0;

  for (std::size_t i = 0; i < root_moves.size(); ++i) {
    check_stop();

    Board child = makeMove(board, root_moves[i]);

    // Threshold-based window: exact scores for moves within threshold of best,
    // fail-low bounds for clearly worse moves.
    int cutoff;
    if (full_window || best_score <= -SCORE_INFINITE + threshold + 1) {
      cutoff = -SCORE_INFINITE;
    } else {
      cutoff = best_score - threshold - 1;
    }
    scores[i] = -negamax(child, depth - 1, -SCORE_INFINITE, -cutoff, 1, KnownVerdict::UNKNOWN);

    if (scores[i] > best_score) {
      best_score = scores[i];
      best_idx = i;
    }
  }

  // Rotate best move to front for better ordering in next iteration
  if (best_idx > 0) {
    Move tmp_move = root_moves[best_idx];
    int tmp_score = scores[best_idx];
    for (std::size_t j = best_idx; j > 0; --j) {
      root_moves[j] = root_moves[j - 1];
      scores[j] = scores[j - 1];
    }
    root_moves[0] = tmp_move;
    scores[0] = tmp_score;
  }

  return best_score;
}

MultiSearchResult Searcher::search_multi(const Board& board, int max_depth,
                                         const TimeControl& tc, int threshold) {
  stats_ = SearchStats{};
  tt_.new_search();
  tc_ = tc;
  tc_.start();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(pos_hash_history_, 0, sizeof(pos_hash_history_));

  MultiSearchResult result;

  MoveList root_moves;
  generateMoves(board, root_moves);
  if (root_moves.empty()) return result;

  // Only one legal move — no decision to make, return immediately
  if (root_moves.size() == 1 && !analyze_mode_) {
    result.moves.push_back({root_moves[0], 0});
    result.depth = 0;
    return result;
  }

  std::vector<int> scores;

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      int best = search_root_all(board, root_moves, depth, threshold, scores);

      // Depth completed — snapshot results
      result.depth = depth;
      result.nodes = stats_.nodes;
      result.tb_hits = stats_.tb_hits;
      result.moves.clear();
      for (std::size_t i = 0; i < root_moves.size(); ++i) {
        result.moves.push_back({root_moves[i], scores[i]});
      }

      if (verbose_) {
        std::cout << ". depth " << depth << " best " << best
                  << " nodes " << stats_.nodes << std::endl;
      }

      if (is_mate_score(best)) break;
      if (tc_.exceeded_soft(stats_.nodes)) break;
    } catch (const SearchInterrupted&) {
      break;  // Use results from previous completed depth
    }
  }

  // Sort by score descending
  std::sort(result.moves.begin(), result.moves.end(),
            [](const MultiSearchResult::MoveScore& a,
               const MultiSearchResult::MoveScore& b) { return a.score > b.score; });

  return result;
}

SearchResult Searcher::secondary_search(const Board& board, MoveList& root_moves,
                                         const SearchResult& primary,
                                         int max_depth) {
  // Check remaining time
  double elapsed = tc_.elapsed_seconds();
  double remaining = tc_.hard_time_seconds - elapsed;
  if (remaining < 0.5) return primary;

  bool is_winning = (primary.score >= SCORE_SPECIAL && primary.score <= SCORE_TB_WIN);

  // Disable WDL probes for the secondary search.
  // DTM probes remain active so the search can find actual mates.
  // RAII guard restores wdl_probe_func_ on all exit paths.
  auto saved_wdl = std::move(wdl_probe_func_);
  wdl_probe_func_ = nullptr;
  struct WDLGuard {
    WDLProbeFunc& dst;
    WDLProbeFunc saved;
    WDLGuard(WDLProbeFunc& d, WDLProbeFunc&& s) : dst(d), saved(std::move(s)) {}
    ~WDLGuard() { dst = std::move(saved); }
  } wdl_guard(wdl_probe_func_, std::move(saved_wdl));

  // Clear TT to purge WDL-based scores
  tt_.clear();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  stats_ = SearchStats{};

  tc_ = TimeControl::with_time(remaining * 0.4, remaining * 0.9);
  tc_.start();

  SearchResult secondary = is_winning ? primary : SearchResult{};
  secondary.phase = is_winning ? SearchPhase::SECONDARY_WINNING : SearchPhase::SECONDARY_LOSING;

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      SearchResult r = search_root(board, root_moves, depth);
      r.depth = depth;
      if (is_winning) {
        // Only adopt the secondary result if it found an actual mate.
        // Otherwise we'll stick with the known WDL winning move.
        if (is_mate_score(r.score)) {
          secondary.best_move = r.best_move;
          secondary.score = r.score;
          secondary.pv = r.pv;
        }
        secondary.nodes += r.nodes;
      } else {
        secondary = r;
        secondary.phase = SearchPhase::SECONDARY_LOSING;
      }
    } catch (const SearchInterrupted&) {
      break;
    }

    if (progress_callback_) {
      SearchResult report = secondary;
      report.depth = depth;
      progress_callback_(report);
    }

    // Found a mate — no need to search deeper
    if (is_winning && is_mate_score(secondary.score)) break;

    if (tc_.exceeded_soft(stats_.nodes)) break;
  }

  // Clear TT after secondary search so scores without WDL
  // don't pollute the next search (which will use WDL probes)
  tt_.clear();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));

  return secondary;
}

SearchResult Searcher::search(const Board& board, int max_depth, const TimeControl& tc) {
  stats_ = SearchStats{};
  tt_.new_search();
  tc_ = tc;
  tc_.start();
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(pos_hash_history_, 0, sizeof(pos_hash_history_));

  SearchResult result;

  // Check if we should use DTM optimal play (≤ tb_piece_limit_ pieces)
  // For wins/losses, DTM gives the optimal move. For draws, fall through
  // to the normal search so the engine can use heuristics to choose.
  int piece_count = std::popcount(board.allPieces());
  if (dtm_manager_ && piece_count <= tb_piece_limit_) {
    Move best_move;
    tablebase::DTM best_dtm;
    if (dtm_manager_->find_best_move(board, best_move, best_dtm)
        && best_dtm != tablebase::DTM_DRAW) {
      result.best_move = best_move;
      result.score = dtm_to_score(best_dtm, 0);
      result.nodes = 1;
      stats_.tb_hits++;
      result.tb_hits = 1;
      return result;
    }
  }

  // Generate root moves once - they will be reordered across iterations
  MoveList root_moves;
  generateMoves(board, root_moves);

  if (root_moves.empty()) {
    result.score = mated_score(0);
    return result;
  }

  // Only one legal move - no need to search, just return it
  if (root_moves.size() == 1 && !analyze_mode_) {
    result.best_move = root_moves[0];
    result.nodes = 0;
    result.depth = 1;

    // Try to get score from tablebase
    if (dtm_manager_ && piece_count <= tb_piece_limit_) {
      tablebase::DTM dtm = dtm_manager_->lookup_dtm(board);
      if (dtm != tablebase::DTM_UNKNOWN) {
        result.score = (dtm == tablebase::DTM_DRAW) ? draw_eval(board)
                                                     : dtm_to_score(dtm, 0);
        stats_.tb_hits++;
        result.tb_hits = 1;
        return result;
      }
    }

    // No tablebase - use eval of resulting position as rough score estimate
    Board child = makeMove(board, root_moves[0]);
    result.score = to_undecided(-eval_(child, 1));
    return result;
  }

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      if (ponder_mode_) {
        std::vector<int> scores;
        search_root_all(board, root_moves, depth, SCORE_INFINITE, scores, true);
        result.best_move = root_moves[0];
        result.score = scores[0];
        result.nodes = stats_.nodes;
        result.tb_hits = stats_.tb_hits;
        result.pv.clear();
        result.pv.push_back(result.best_move);
        Board child = makeMove(board, result.best_move);
        std::vector<Move> continuation;
        extract_pv(child, continuation, depth - 1);
        result.pv.insert(result.pv.end(), continuation.begin(), continuation.end());
        // Populate root_scores for all moves, sorted best to worst
        result.root_scores.clear();
        for (size_t i = 0; i < root_moves.size(); ++i) {
          result.root_scores.emplace_back(root_moves[i], scores[i]);
        }
        std::sort(result.root_scores.begin(), result.root_scores.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
      } else {
        result = search_root(board, root_moves, depth);
      }
      result.depth = depth;
    } catch (const SearchInterrupted&) {
      break;  // Return best result from previous depth
    }

    // Call progress callback if set
    if (progress_callback_) {
      progress_callback_(result);
    }

    if (verbose_) {
      int display_score = result.score;
      std::string score_label;
      if (is_proven_draw(result.score)) {
        score_label = "draw(";
        double val = result.score / 100.0;
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%+.2f", val);
        score_label += buf;
        score_label += ")";
      } else if (!is_special_score(result.score)) {
        display_score = undecided_to_display(result.score);
      }
      std::cout << ". depth " << depth;
      if (!score_label.empty()) {
        std::cout << " score " << score_label;
      } else {
        std::cout << " score " << display_score;
      }
      std::cout << " nodes " << result.nodes;
      if (result.tb_hits > 0) {
        std::cout << " tbhits " << result.tb_hits;
      }
      if (!result.pv.empty()) {
        std::cout << " pv";
        Board pos = board;
        bool white_view = white_perspective_;
        for (const Move& m : result.pv) {
          // Find from/to squares
          int from = __builtin_ctz(m.from_xor_to & pos.white);
          if (from >= 32) from = __builtin_ctz(m.from_xor_to & pos.black);
          int to = __builtin_ctz(m.from_xor_to ^ (1u << from));
          // Flip squares for black's perspective
          int disp_from = white_view ? (from + 1) : (32 - from);
          int disp_to = white_view ? (to + 1) : (32 - to);
          std::cout << " " << disp_from << (m.isCapture() ? "x" : "-") << disp_to;
          pos = makeMove(pos, m);
          white_view = !white_view;  // Alternate perspective for each ply
        }
      }
      std::cout << std::endl;
    }

    // Early exit if we found a forced mate or forced move
    if (is_mate_score(result.score) || result.nodes == 0) {
      break;
    }

    // WDL win/loss score detected — switch to secondary search immediately
    // (proven draws are handled by depth reduction in probe_wdl, no secondary search needed)
    // Skip during pondering: secondary search is expensive and the result won't be used directly
    // Skip when draw_value_ != 0 (Espada/Broquel mode): secondary search disables WDL probes,
    // but in these modes WDL probes are essential because draws are remapped to wins/losses.
    // Without them the search can't distinguish drawing from losing moves.
    if (wdl_probe_func_ && !ponder_mode_ && draw_value_ == 0) {
      bool is_wdl_winloss = (result.score <= -SCORE_SPECIAL) ||
                             (result.score >= SCORE_SPECIAL && result.score <= SCORE_TB_WIN);
      if (is_wdl_winloss) {
        result = secondary_search(board, root_moves, result, max_depth);
        break;
      }
    }

    // Stop if we've exceeded any soft limit
    if (tc_.exceeded_soft(result.nodes)) {
      break;
    }
  }

  return result;
}

} // namespace search
