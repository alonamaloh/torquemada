#include "search.hpp"
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>

namespace search {

// Temperature values for variety modes
constexpr double TEMPERATURE_CURIOUS = 273.0;
constexpr double TEMPERATURE_SAFE = TEMPERATURE_CURIOUS / 9.0;
constexpr double TEMPERATURE_WILD = TEMPERATURE_CURIOUS * 3.0;

// Score threshold multiplier: threshold = T * ln(10) where moves at threshold
// have 10% probability of being selected compared to the best move
constexpr double THRESHOLD_MULTIPLIER = 2.302585;  // ln(10)

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
                   const std::string& nn_model_path, const std::string& dtm_nn_model_path)
    : tt_(64), eval_(random_eval), tb_piece_limit_(tb_piece_limit) {
  if (!tb_directory.empty() && tb_piece_limit > 0) {
    dtm_manager_owned_ = std::make_unique<tablebase::DTMTablebaseManager>(tb_directory);
    dtm_manager_owned_->preload(tb_piece_limit);
    dtm_manager_ = dtm_manager_owned_.get();
  }

  // Load neural networks if paths provided
  if (!nn_model_path.empty()) {
    nn_model_ = std::make_unique<nn::MLP>(nn_model_path);
  }
  if (!dtm_nn_model_path.empty()) {
    dtm_nn_model_ = std::make_unique<nn::MLP>(dtm_nn_model_path);
  }

  // Set up evaluation function based on available models
  if (nn_model_ && dtm_nn_model_) {
    // Use endgame model for 6-7 pieces, general model for 8+
    eval_ = [this](const Board& board, int ply) {
      int piece_count = std::popcount(board.allPieces());
      if (piece_count <= 7) {
        return dtm_nn_model_->evaluate(board, effective_draw_score(ply));
      }
      return nn_model_->evaluate(board, effective_draw_score(ply));
    };
  } else if (nn_model_) {
    eval_ = [this](const Board& board, int ply) {
      return nn_model_->evaluate(board, effective_draw_score(ply));
    };
  } else if (dtm_nn_model_) {
    eval_ = [this](const Board& board, int ply) {
      return dtm_nn_model_->evaluate(board, effective_draw_score(ply));
    };
  }
}

Searcher::Searcher(const tablebase::DTMTablebaseManager* dtm_tb, int tb_piece_limit,
                   const std::string& nn_model_path, const std::string& dtm_nn_model_path)
    : tt_(64), eval_(random_eval), dtm_manager_(dtm_tb), tb_piece_limit_(tb_piece_limit) {
  // Load neural networks if paths provided
  if (!nn_model_path.empty()) {
    nn_model_ = std::make_unique<nn::MLP>(nn_model_path);
  }
  if (!dtm_nn_model_path.empty()) {
    dtm_nn_model_ = std::make_unique<nn::MLP>(dtm_nn_model_path);
  }

  // Set up evaluation function based on available models
  if (nn_model_ && dtm_nn_model_) {
    eval_ = [this](const Board& board, int ply) {
      int piece_count = std::popcount(board.allPieces());
      if (piece_count <= 7) {
        return dtm_nn_model_->evaluate(board, effective_draw_score(ply));
      }
      return nn_model_->evaluate(board, effective_draw_score(ply));
    };
  } else if (nn_model_) {
    eval_ = [this](const Board& board, int ply) {
      return nn_model_->evaluate(board, effective_draw_score(ply));
    };
  } else if (dtm_nn_model_) {
    eval_ = [this](const Board& board, int ply) {
      return dtm_nn_model_->evaluate(board, effective_draw_score(ply));
    };
  }
}

Searcher::~Searcher() = default;

int Searcher::dtm_to_score(tablebase::DTM dtm, int ply) {
  if (dtm == tablebase::DTM_UNKNOWN) {
    return 0;  // Shouldn't happen
  }
  if (dtm == tablebase::DTM_DRAW) {
    return effective_draw_score(ply);
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

  // Only probe when n_reversible == 0 (after a capture or pawn move)
  // This forces the winning side to find moves that make progress
  if (board.n_reversible != 0) return false;

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

  stats_.tb_hits++;
  score = dtm_to_score(dtm, ply);
  return true;
}

int Searcher::negamax(const Board& board, int depth, int alpha, int beta, int ply) {
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
        // Found a repetition - return draw score
        return effective_draw_score(ply);
      }
    }
  }

  // Check for tablebase hit (before TT to get exact values)
  int tb_score;
  if (probe_tb(board, ply, tb_score)) {
    return tb_score;
  }

  // Probe transposition table (using position hash without n_reversible)
  std::uint64_t key = pos_hash;
  TTEntry tt_entry;
  CompactMove tt_compact_move = 0;

  if (tt_.probe(key, tt_entry)) {
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

  // Leaf node: only evaluate when depth <= 0 AND no captures available
  // If captures exist, continue searching (quiescence)
  bool has_captures = moves[0].isCapture();
  if (depth <= 0 && !has_captures) {
    int score = eval_(board, ply);
    // Scale score towards draw based on n_reversible to encourage progress
    // The more reversible moves without progress, the smaller the score
    // Scale factor: (256 - n_reversible) / 256, capped at 50 reversible moves
    if (board.n_reversible > 0 && !is_special_score(score)) {
      int scale = std::max(256 - static_cast<int>(board.n_reversible) * 4, 56);  // Min 56/256 ~= 22%
      int draw = effective_draw_score(ply);
      score = draw + (score - draw) * scale / 256;
    }
    return score;
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
  TTFlag flag = TTFlag::UPPER_BOUND;
  Move first_move = moves[0];
  bool is_first = true;

  for (const Move& move : moves) {
    Board child = makeMove(board, move);
    int score;

    if (is_first) {
      // Search first move with full window
      score = -negamax(child, depth - 1, -beta, -alpha, ply + 1);
    } else {
      // LMR: reduce depth for late non-capture moves when depth is sufficient
      int reduction = (depth >= 3 && !move.isCapture()) ? 1 : 0;
      // PVS: try null-window search first, possibly at reduced depth
      score = -negamax(child, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
      // If it fails high, re-search with full window and full depth
      if (score > alpha) {
        score = -negamax(child, depth - 1, -beta, -alpha, ply + 1);
      }
    }

    if (score > best_score) {
      best_score = score;
      best_move = move;

      if (score > alpha) {
        alpha = score;
        flag = TTFlag::EXACT;

        if (alpha >= beta) {
          flag = TTFlag::LOWER_BOUND;
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

  // Store in transposition table
  // When n_reversible > 0, store with a trivial bound (>= -INF) so the best move
  // is preserved for ordering, but the score won't cause any cutoffs.
  // This avoids search instability from history-dependent effects.
  if (board.n_reversible == 0) {
    // For special scores (mate/TB), only store bounds, not exact values,
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
    int score = -negamax(child, depth - 1, -beta, -alpha, 1);

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

SearchResult Searcher::search_root_variety(const Board& board, MoveList& moves, int depth) {
  SearchResult result;
  result.depth = depth;

  // Safety check - should never happen but guard against it
  if (moves.empty()) {
    result.score = mated_score(0);
    return result;
  }

  // Store root position hash for repetition detection
  pos_hash_history_[0] = board.position_hash();

  // Get temperature based on variety mode
  double temperature;
  switch (variety_mode_) {
    case VarietyMode::SAFE:
      temperature = TEMPERATURE_SAFE;
      break;
    case VarietyMode::WILD:
      temperature = TEMPERATURE_WILD;
      break;
    case VarietyMode::CURIOUS:
    default:
      temperature = TEMPERATURE_CURIOUS;
      break;
  }

  // Compute score threshold: moves at this threshold have 10% probability of best
  int variety_threshold = static_cast<int>(temperature * THRESHOLD_MULTIPLIER + 0.5);

  // Structure to hold candidate moves with their scores
  struct Candidate {
    Move move;
    int score;
  };
  std::vector<Candidate> candidates;
  candidates.reserve(moves.size());

  // Step 1: Search first move with full window to establish baseline
  int best_score = -SCORE_INFINITE;
  try {
    const Move& move = moves[0];
    Board child = makeMove(board, move);
    int score = -negamax(child, depth - 1, -SCORE_INFINITE, SCORE_INFINITE, 1);
    best_score = score;
    candidates.push_back({move, score});
  } catch (const SearchInterrupted&) {
    // If interrupted on first move, just return it without a score
    result.best_move = moves[0];
    result.score = 0;
    result.nodes = stats_.nodes;
    result.tb_hits = stats_.tb_hits;
    return result;
  }

  // Step 2: For remaining moves, use null-window search at threshold
  // If a move might be within variety_threshold of best, re-search with full window
  int threshold = best_score - variety_threshold;
  for (std::size_t i = 1; i < moves.size(); ++i) {
    try {
      const Move& move = moves[i];
      Board child = makeMove(board, move);

      // Null-window search: check if score > threshold
      int score = -negamax(child, depth - 1, -threshold - 1, -threshold, 1);

      if (score > threshold) {
        // Move might be good enough - re-search with full window
        score = -negamax(child, depth - 1, -SCORE_INFINITE, SCORE_INFINITE, 1);
        candidates.push_back({move, score});

        // Update best_score and threshold if we found a better move
        if (score > best_score) {
          best_score = score;
          threshold = best_score - variety_threshold;
        }
      }
    } catch (const SearchInterrupted&) {
      // If interrupted, stop searching more moves and use what we have
      break;
    }
  }

  // Step 3: Filter candidates to only include moves within threshold of best
  std::vector<Candidate> valid_candidates;
  for (const auto& c : candidates) {
    if (c.score >= best_score - variety_threshold) {
      valid_candidates.push_back(c);
    }
  }

  // Step 4: Select move using softmax sampling
  Move selected_move;
  int selected_score;

  if (valid_candidates.size() <= 1) {
    // Only one candidate (or none) - no variety needed
    selected_move = valid_candidates.empty() ? moves[0] : valid_candidates[0].move;
    selected_score = valid_candidates.empty() ? best_score : valid_candidates[0].score;
  } else {
    // Compute softmax probabilities (subtract max for numerical stability)
    std::vector<double> weights;
    weights.reserve(valid_candidates.size());
    double max_score = static_cast<double>(best_score);
    for (const auto& c : valid_candidates) {
      double w = std::exp((static_cast<double>(c.score) - max_score) / temperature);
      weights.push_back(w);
    }

    // Compute cumulative distribution
    double total = 0.0;
    for (double w : weights) {
      total += w;
    }

    // Sample using RNG
    RandomBits* rng = rng_;
    if (!rng) {
      if (!owned_rng_) {
        owned_rng_ = std::make_unique<RandomBits>(
            std::chrono::steady_clock::now().time_since_epoch().count());
      }
      rng = owned_rng_.get();
    }

    double r = (static_cast<double>((*rng)()) / static_cast<double>(RandomBits::max())) * total;
    double cumulative = 0.0;
    std::size_t selected_idx = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
      cumulative += weights[i];
      if (r < cumulative) {
        selected_idx = i;
        break;
      }
    }

    selected_move = valid_candidates[selected_idx].move;
    selected_score = valid_candidates[selected_idx].score;

    // Populate variety_candidates for debugging
    for (std::size_t i = 0; i < valid_candidates.size(); ++i) {
      VarietyCandidate vc;
      vc.move = valid_candidates[i].move;
      vc.score = valid_candidates[i].score;
      vc.probability = weights[i] / total;
      vc.selected = (i == selected_idx);
      result.variety_candidates.push_back(vc);
    }

    if (verbose_ && valid_candidates.size() > 1) {
      std::cout << ". variety: " << valid_candidates.size() << " candidates";
      for (std::size_t i = 0; i < valid_candidates.size(); ++i) {
        double prob = weights[i] / total * 100.0;
        std::cout << (i == 0 ? " [" : ", ");
        std::cout << valid_candidates[i].score;
        if (i == selected_idx) std::cout << "*";
        std::cout << " " << static_cast<int>(prob + 0.5) << "%";
      }
      std::cout << "]" << std::endl;
    }
  }

  result.best_move = selected_move;
  result.score = selected_score;
  result.nodes = stats_.nodes;
  result.tb_hits = stats_.tb_hits;

  // Rotate selected move to front for next iteration's move ordering
  for (std::size_t i = 0; i < moves.size(); ++i) {
    if (moves[i].from_xor_to == selected_move.from_xor_to &&
        moves[i].captures == selected_move.captures) {
      if (i > 0) {
        std::rotate(moves.begin(), moves.begin() + i, moves.begin() + i + 1);
      }
      break;
    }
  }

  // Extract PV from TT
  if (result.best_move.from_xor_to != 0) {
    result.pv.push_back(result.best_move);
    Board child = makeMove(board, result.best_move);
    std::vector<Move> continuation;
    extract_pv(child, continuation, depth - 1);
    result.pv.insert(result.pv.end(), continuation.begin(), continuation.end());
  }

  return result;
}

SearchResult Searcher::search(const Board& board, int max_depth, std::uint64_t max_nodes,
                              int game_ply) {
  stats_ = SearchStats{};
  tt_.new_search();
  hard_node_limit_ = (max_nodes > 0) ? max_nodes * 5 : 0;
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));
  std::memset(pos_hash_history_, 0, sizeof(pos_hash_history_));

  SearchResult result;

  // Check if we should use DTM optimal play (≤ tb_piece_limit_ pieces)
  int piece_count = std::popcount(board.allPieces());
  if (dtm_manager_ && piece_count <= tb_piece_limit_) {
    Move best_move;
    tablebase::DTM best_dtm;
    if (dtm_manager_->find_best_move(board, best_move, best_dtm)) {
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
  if (root_moves.size() == 1) {
    result.best_move = root_moves[0];
    result.nodes = 0;
    result.depth = 1;

    // Try to get score from tablebase
    if (dtm_manager_ && piece_count <= tb_piece_limit_) {
      tablebase::DTM dtm = dtm_manager_->lookup_dtm(board);
      if (dtm != tablebase::DTM_UNKNOWN) {
        result.score = dtm_to_score(dtm, 0);
        stats_.tb_hits++;
        result.tb_hits = 1;
        return result;
      }
    }

    // No tablebase - use eval of resulting position as rough score estimate
    Board child = makeMove(board, root_moves[0]);
    result.score = -eval_(child, 1);
    return result;
  }

  // Determine if we should use variety search
  // Only in the first 10 moves (game_ply < 20) and when variety mode is enabled
  bool use_variety = (variety_mode_ != VarietyMode::NONE) && (game_ply < 20);
  bool variety_applied = false;

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      result = search_root(board, root_moves, depth);
      result.depth = depth;
    } catch (const SearchInterrupted&) {
      break;  // Return best result from previous depth
    }

    // Call progress callback if set
    if (progress_callback_) {
      progress_callback_(result);
    }

    if (verbose_) {
      std::cout << ". depth " << depth << " score " << result.score
                << " nodes " << result.nodes;
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

    // Check if we've reached the variety threshold (25% of max_nodes)
    // If so, apply variety search and finish
    if (use_variety && !variety_applied && max_nodes > 0 && result.nodes >= max_nodes / 4) {
      // This is effectively the last depth - apply variety selection
      if (verbose_) {
        std::cout << ". variety search at depth " << (depth + 1) << std::endl;
      }
      try {
        result = search_root_variety(board, root_moves, depth + 1);
        result.depth = depth + 1;
        variety_applied = true;
        if (verbose_) {
          std::cout << ". variety search completed, score=" << result.score << std::endl;
        }
      } catch (const SearchInterrupted&) {
        // If interrupted during variety search, use previous result
        if (verbose_) {
          std::cout << ". variety search interrupted, using previous result" << std::endl;
        }
      } catch (...) {
        // Catch any other exception - fall back to regular search
        if (verbose_) {
          std::cout << ". variety search threw exception, falling back to regular search" << std::endl;
        }
        try {
          result = search_root(board, root_moves, depth);
          result.depth = depth;
        } catch (const SearchInterrupted&) {
          // If regular search also interrupted, use whatever result we had
        }
      }
      break;
    }

    // Stop if we've exceeded the soft node limit
    if (max_nodes > 0 && result.nodes >= max_nodes) {
      break;
    }
  }

  hard_node_limit_ = 0;
  return result;
}

} // namespace search
