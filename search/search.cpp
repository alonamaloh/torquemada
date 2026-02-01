#include "search.hpp"
#include <algorithm>
#include <bit>
#include <cstring>
#include <iostream>

namespace search {

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

Searcher::Searcher(const std::string& tb_directory, int tb_piece_limit, int dtm_piece_limit,
                   const std::string& nn_model_path, const std::string& dtm_nn_model_path)
    : tt_(64), eval_(random_eval), tb_piece_limit_(tb_piece_limit),
      dtm_piece_limit_(dtm_piece_limit) {
  if (!tb_directory.empty()) {
    tb_manager_owned_ = std::make_unique<CompressedTablebaseManager>(tb_directory);
    dtm_manager_owned_ = std::make_unique<tablebase::DTMTablebaseManager>(tb_directory);
    // Preload for thread-safe const access
    tb_manager_owned_->preload(tb_piece_limit);
    dtm_manager_owned_->preload(dtm_piece_limit);
    tb_manager_ = tb_manager_owned_.get();
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

Searcher::Searcher(const CompressedTablebaseManager* wdl_tb, const tablebase::DTMTablebaseManager* dtm_tb,
                   int tb_piece_limit, int dtm_piece_limit, const std::string& nn_model_path,
                   const std::string& dtm_nn_model_path)
    : tt_(64), eval_(random_eval), tb_manager_(wdl_tb), dtm_manager_(dtm_tb),
      tb_piece_limit_(tb_piece_limit), dtm_piece_limit_(dtm_piece_limit) {
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
  if (!tb_manager_) return false;

  // Only probe when n_reversible == 0 (after a capture or pawn move)
  // This forces the winning side to find moves that make progress
  if (board.n_reversible != 0) return false;

  int piece_count = std::popcount(board.allPieces());
  if (piece_count > tb_piece_limit_) return false;

  Value result = tb_manager_->lookup_wdl_preloaded(board);
  if (result == Value::UNKNOWN) return false;

  stats_.tb_hits++;

  // Adjust TB score by ply so we prefer shorter paths to wins
  switch (result) {
    case Value::WIN:
      score = SCORE_TB_WIN - ply;
      break;
    case Value::LOSS:
      score = SCORE_TB_LOSS + ply;
      break;
    case Value::DRAW:
      score = effective_draw_score(ply);
      break;
    default:
      return false;
  }
  return true;
}

int Searcher::negamax(const Board& board, int depth, int alpha, int beta, int ply) {
  check_stop();  // Throws SearchInterrupted if we should stop

  stats_.nodes++;

  // Check for tablebase hit (before TT to get exact values)
  int tb_score;
  if (probe_tb(board, ply, tb_score)) {
    return tb_score;
  }

  // Probe transposition table
  std::uint64_t key = board.hash();
  TTEntry tt_entry;
  CompactMove tt_compact_move = 0;

  if (tt_.probe(key, tt_entry)) {
    stats_.tt_hits++;
    tt_compact_move = tt_entry.best_move;

    // Can we use the TT score?
    if (tt_entry.depth >= depth) {
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
    return eval_(board, ply);
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
    int score = -negamax(child, depth - 1, -beta, -alpha, ply + 1);

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
  // For special scores (mate/TB), only store bounds, not exact values,
  // because these scores are relative to the root and may not be valid
  // when accessed from a different search path
  TTFlag store_flag = flag;
  if (is_special_score(best_score) && flag == TTFlag::EXACT) {
    // Convert exact to the appropriate bound
    store_flag = (best_score > 0) ? TTFlag::LOWER_BOUND : TTFlag::UPPER_BOUND;
  }
  tt_.store(key, best_score, depth, store_flag, best_move);

  return best_score;
}

void Searcher::extract_pv(const Board& board, std::vector<Move>& pv, int max_depth) {
  pv.clear();
  Board pos = board;
  std::uint64_t seen_keys[64];  // To detect cycles
  int seen_count = 0;

  for (int i = 0; i < max_depth && seen_count < 64; ++i) {
    std::uint64_t key = pos.hash();

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

SearchResult Searcher::search(const Board& board, int max_depth, std::uint64_t max_nodes) {
  stats_ = SearchStats{};
  tt_.new_search();
  hard_node_limit_ = (max_nodes > 0) ? max_nodes * 5 : 0;
  std::memset(killers_, 0, sizeof(killers_));
  std::memset(history_, 0, sizeof(history_));

  SearchResult result;

  // Check if we should use DTM optimal play (≤ dtm_piece_limit pieces)
  int piece_count = std::popcount(board.allPieces());
  if (dtm_manager_ && piece_count <= dtm_piece_limit_) {
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
    if (dtm_manager_ && piece_count <= dtm_piece_limit_) {
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

  for (int depth = 1; depth <= max_depth; ++depth) {
    try {
      result = search_root(board, root_moves, depth);
      result.depth = depth;
    } catch (const SearchInterrupted&) {
      break;  // Return best result from previous depth
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

    // Stop if we've exceeded the soft node limit
    if (max_nodes > 0 && result.nodes >= max_nodes) {
      break;
    }

    // Early exit if we found a forced mate or forced move
    if (is_mate_score(result.score) || result.nodes == 0) {
      break;
    }
  }

  hard_node_limit_ = 0;
  return result;
}

} // namespace search
