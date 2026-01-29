#include "search.h"
#include <algorithm>
#include <bit>
#include <iostream>

namespace search {

// Piece-square tables for positional evaluation
// Bonus for pawn advancement (row 0-7, where 7 is promotion rank for white)
constexpr int PAWN_ROW_BONUS[8] = {0, 0, 2, 5, 10, 20, 35, 0};  // Row 7 = queen, handled separately

// Center squares are slightly better for queens
constexpr Bb CENTER_MASK = 0x00666600u;  // Squares 9,10,13,14,17,18,21,22

// Material + positional evaluation
int material_eval(const Board& board) {
  constexpr int PAWN_VALUE = 100;
  constexpr int QUEEN_VALUE = 300;
  constexpr int CENTER_BONUS = 5;

  int score = 0;

  // Material
  score += std::popcount(board.whitePawns()) * PAWN_VALUE;
  score += std::popcount(board.whiteQueens()) * QUEEN_VALUE;
  score -= std::popcount(board.blackPawns()) * PAWN_VALUE;
  score -= std::popcount(board.blackQueens()) * QUEEN_VALUE;

  // Pawn advancement bonus (white pawns want to go to higher rows)
  for (Bb pawns = board.whitePawns(); pawns; pawns &= pawns - 1) {
    int sq = std::countr_zero(pawns);
    int row = sq / 4;
    score += PAWN_ROW_BONUS[row];
  }

  // Black pawns want to go to lower rows (from their perspective after flip)
  // But board is stored with white to move, so black pawns on low rows are advanced
  for (Bb pawns = board.blackPawns(); pawns; pawns &= pawns - 1) {
    int sq = std::countr_zero(pawns);
    int row = sq / 4;
    score -= PAWN_ROW_BONUS[7 - row];  // Flip row for black
  }

  // Queen centralization
  score += std::popcount(board.whiteQueens() & CENTER_MASK) * CENTER_BONUS;
  score -= std::popcount(board.blackQueens() & CENTER_MASK) * CENTER_BONUS;

  return score;
}

// Hash-based evaluation: reproducible pseudo-random score derived from position hash
// Returns a score in the range [-500, +500] based on the hash
int hash_eval(const Board& board) {
  std::uint64_t h = board.hash();

  // Mix the hash to get good distribution
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;

  // Convert to signed score in range [-500, +500]
  // Use the lower 32 bits, treat as signed, then scale
  std::int32_t raw = static_cast<std::int32_t>(h & 0xFFFFFFFFu);
  return raw % 501;  // Range [-500, +500]
}

Searcher::Searcher(const std::string& tb_directory, int tb_piece_limit)
    : tt_(64), eval_(material_eval), tb_piece_limit_(tb_piece_limit) {
  if (!tb_directory.empty()) {
    tb_manager_ = std::make_unique<CompressedTablebaseManager>(tb_directory);
  }
}

Searcher::~Searcher() = default;

bool Searcher::probe_tb(const Board& board, int ply, int& score) {
  if (!tb_manager_) return false;

  // Only probe when n_reversible == 0 (after a capture or pawn move)
  // This forces the winning side to find moves that make progress
  if (board.n_reversible != 0) return false;

  int piece_count = std::popcount(board.allPieces());
  if (piece_count > tb_piece_limit_) return false;

  Value result = tb_manager_->lookup_wdl(board);
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
      score = SCORE_DRAW;
      break;
    default:
      return false;
  }
  return true;
}

void Searcher::order_moves(std::vector<Move>& moves, const Board& /*board*/, const Move& tt_move) {
  // Simple move ordering:
  // 1. TT move first (if valid)
  // 2. Captures before quiet moves (captures are mandatory, but within same length)

  auto score_move = [&](const Move& m) -> int {
    if (m == tt_move) return 10000;  // TT move first
    if (m.isCapture()) return 1000 + std::popcount(m.captures);  // More captures = better
    return 0;
  };

  std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
    return score_move(a) > score_move(b);
  });
}

int Searcher::quiescence(const Board& board, int alpha, int beta, int ply) {
  stats_.qnodes++;
  stats_.sel_depth = std::max(stats_.sel_depth, ply);

  // Check for tablebase hit
  int tb_score;
  if (probe_tb(board, ply, tb_score)) {
    return tb_score;
  }

  // Generate moves
  std::vector<Move> moves;
  generateMoves(board, moves);

  // Terminal node - no moves means loss (we're stalemated/captured out)
  if (moves.empty()) {
    return mated_score(ply);
  }

  // If no captures available, this is a quiet position - evaluate
  if (!moves[0].isCapture()) {
    return eval_(board);
  }

  // We have captures - must continue searching
  // In Spanish checkers, captures are mandatory, so we search all of them

  int best_score = -SCORE_INFINITE;

  for (const Move& move : moves) {
    Board child = makeMove(board, move);
    int score = -quiescence(child, -beta, -alpha, ply + 1);

    if (score > best_score) {
      best_score = score;
      if (score > alpha) {
        alpha = score;
        if (alpha >= beta) {
          break;  // Beta cutoff
        }
      }
    }
  }

  return best_score;
}

int Searcher::negamax(const Board& board, int depth, int alpha, int beta, int ply) {
  stats_.nodes++;

  // Check for tablebase hit (before TT to get exact values)
  int tb_score;
  if (probe_tb(board, ply, tb_score)) {
    return tb_score;
  }

  // Probe transposition table
  std::uint64_t key = board.hash();
  TTEntry tt_entry;
  Move tt_move;

  if (tt_.probe(key, tt_entry)) {
    stats_.tt_hits++;
    tt_move = tt_entry.best_move;

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
  std::vector<Move> moves;
  generateMoves(board, moves);

  // Terminal node - no moves means loss
  if (moves.empty()) {
    return mated_score(ply);
  }

  // Leaf node or captures available - go to quiescence
  // We only use static eval when position is quiet (no captures)
  if (depth <= 0) {
    if (moves[0].isCapture()) {
      // Captures available - continue with quiescence
      return quiescence(board, alpha, beta, ply);
    } else {
      // Quiet position - evaluate
      return eval_(board);
    }
  }

  // Order moves for better pruning
  order_moves(moves, board, tt_move);

  int best_score = -SCORE_INFINITE;
  Move best_move;
  TTFlag flag = TTFlag::UPPER_BOUND;

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
          break;  // Beta cutoff
        }
      }
    }
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

SearchResult Searcher::search(const Board& board, int depth) {
  stats_ = SearchStats{};

  SearchResult result;
  result.depth = depth;

  // Generate root moves
  std::vector<Move> moves;
  generateMoves(board, moves);

  if (moves.empty()) {
    result.score = mated_score(0);
    return result;
  }

  int alpha = -SCORE_INFINITE;
  int beta = SCORE_INFINITE;

  for (const Move& move : moves) {
    Board child = makeMove(board, move);
    int score = -negamax(child, depth - 1, -beta, -alpha, 1);

    if (score > alpha) {
      alpha = score;
      result.best_move = move;
      result.score = score;
    }
  }

  result.nodes = stats_.nodes + stats_.qnodes;
  result.tb_hits = stats_.tb_hits;
  return result;
}

SearchResult Searcher::search_iterative(const Board& board, int max_depth) {
  SearchResult result;

  for (int depth = 1; depth <= max_depth; ++depth) {
    result = search(board, depth);
    result.depth = depth;

    // Early exit if we found a forced mate
    if (is_mate_score(result.score)) {
      break;
    }
  }

  return result;
}

} // namespace search
