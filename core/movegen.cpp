#include "movegen.hpp"
#include <algorithm>
#include <bit>

namespace {

// Collector for compact moves (no path tracking)
struct MoveCollector {
  static constexpr bool tracks_path = false;
  MoveList& moves;
  explicit MoveCollector(MoveList& m) : moves(m) {}
  void add(const Move& m) { moves.push_back(m); }
  void add(const Move& m, const std::vector<int>&) { moves.push_back(m); }
};

// Collector for full moves with path info
struct FullMoveCollector {
  static constexpr bool tracks_path = true;
  std::vector<FullMove>& moves;
  explicit FullMoveCollector(std::vector<FullMove>& m) : moves(m) {}
  void add(const Move& m, const std::vector<int>& path) {
    moves.emplace_back(m, path);
  }
};

// Templated pawn capture generator
template<typename Collector>
struct PawnCaptureGen {
  Bb canCaptureNW;
  Bb canCaptureNE;
  Collector& collector;

  void generate(Bb from, Bb to, Bb captures, std::vector<int>& path) {
    if constexpr (Collector::tracks_path) {
      path.push_back(std::countr_zero(to));
    }

    bool continued = false;
    if (to & canCaptureNW) {
      continued = true;
      generate(from, moveNW(moveNW(to)), captures | moveNW(to), path);
    }
    if (to & canCaptureNE) {
      continued = true;
      generate(from, moveNE(moveNE(to)), captures | moveNE(to), path);
    }

    if (!continued) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, captures), path);
      } else {
        collector.add(Move(from ^ to, captures));
      }
    }

    if constexpr (Collector::tracks_path) {
      path.pop_back();
    }
  }
};

// Templated queen capture generator
template<typename Collector>
struct QueenCaptureGen {
  Bb canCaptureNW, canCaptureNE, canCaptureSE, canCaptureSW;
  Bb black;
  Bb empty;
  Collector& collector;

  void generate(Bb from, Bb to, Bb captures, std::vector<int>& path) {
    if constexpr (Collector::tracks_path) {
      path.push_back(std::countr_zero(to));
    }
    Bb capturable = canCaptureNW | canCaptureNE | canCaptureSE | canCaptureSW;

    bool continued = false;
    if (to & capturable) {
      if (to & canCaptureNW) {
        Bb cap = to;
        while (cap = moveNW(cap), cap & empty);
        if (cap & black & ~captures)
          for (Bb newTo = moveNW(cap); newTo & empty; newTo = moveNW(newTo)) {
            continued = true;
            generate(from, newTo, captures | cap, path);
          }
      }
      if (to & canCaptureNE) {
        Bb cap = to;
        while (cap = moveNE(cap), cap & empty);
        if (cap & black & ~captures)
          for (Bb newTo = moveNE(cap); newTo & empty; newTo = moveNE(newTo)) {
            continued = true;
            generate(from, newTo, captures | cap, path);
          }
      }
      if (to & canCaptureSE) {
        Bb cap = to;
        while (cap = moveSE(cap), cap & empty);
        if (cap & black & ~captures)
          for (Bb newTo = moveSE(cap); newTo & empty; newTo = moveSE(newTo)) {
            continued = true;
            generate(from, newTo, captures | cap, path);
          }
      }
      if (to & canCaptureSW) {
        Bb cap = to;
        while (cap = moveSW(cap), cap & empty);
        if (cap & black & ~captures)
          for (Bb newTo = moveSW(cap); newTo & empty; newTo = moveSW(newTo)) {
            continued = true;
            generate(from, newTo, captures | cap, path);
          }
      }
    }

    if (!continued) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, captures), path);
      } else {
        collector.add(Move(from ^ to, captures));
      }
    }

    if constexpr (Collector::tracks_path) {
      path.pop_back();
    }
  }
};

// Core move generation logic, templated on collector type
template<typename Collector>
void generateMovesImpl(const Board& board, Collector& collector) {
  Bb empty = board.empty();
  std::vector<int> path;  // Only used if Collector::tracks_path

  // === Pawn captures ===
  Bb pawnCanCaptureNW = moveSE(board.black & moveSE(empty));
  Bb pawnCanCaptureNE = moveSW(board.black & moveSW(empty));

  if (board.whitePawns() & (pawnCanCaptureNW | pawnCanCaptureNE)) {
    PawnCaptureGen<Collector> gen{pawnCanCaptureNW, pawnCanCaptureNE, collector};

    for (Bb x = board.whitePawns() & pawnCanCaptureNW; x; x &= x - 1) {
      Bb from = x & -x;
      if constexpr (Collector::tracks_path) {
        path = {std::countr_zero(from)};
      }
      gen.generate(from, moveNW(moveNW(from)), moveNW(from), path);
    }
    for (Bb x = board.whitePawns() & pawnCanCaptureNE; x; x &= x - 1) {
      Bb from = x & -x;
      if constexpr (Collector::tracks_path) {
        path = {std::countr_zero(from)};
      }
      gen.generate(from, moveNE(moveNE(from)), moveNE(from), path);
    }
  }

  // === Queen captures ===
  Bb kingOrEmpty = board.whiteQueens() | empty;
  Bb canBeCapturedNW = board.black & moveSE(kingOrEmpty);
  Bb canBeCapturedNE = board.black & moveSW(kingOrEmpty);
  Bb canBeCapturedSE = board.black & moveNW(kingOrEmpty);
  Bb canBeCapturedSW = board.black & moveNE(kingOrEmpty);

  Bb queenCanCaptureNW = 0;
  for (Bb x = moveSE(canBeCapturedNW) & kingOrEmpty; x; x = moveSE(x) & kingOrEmpty) {
    queenCanCaptureNW |= x;
    x &= empty;
  }
  Bb queenCanCaptureNE = 0;
  for (Bb x = moveSW(canBeCapturedNE) & kingOrEmpty; x; x = moveSW(x) & kingOrEmpty) {
    queenCanCaptureNE |= x;
    x &= empty;
  }
  Bb queenCanCaptureSE = 0;
  for (Bb x = moveNW(canBeCapturedSE) & kingOrEmpty; x; x = moveNW(x) & kingOrEmpty) {
    queenCanCaptureSE |= x;
    x &= empty;
  }
  Bb queenCanCaptureSW = 0;
  for (Bb x = moveNE(canBeCapturedSW) & kingOrEmpty; x; x = moveNE(x) & kingOrEmpty) {
    queenCanCaptureSW |= x;
    x &= empty;
  }

  Bb queenCapturable = queenCanCaptureNW | queenCanCaptureNE | queenCanCaptureSE | queenCanCaptureSW;

  if (board.whiteQueens() & queenCapturable) {
    QueenCaptureGen<Collector> gen{queenCanCaptureNW, queenCanCaptureNE, queenCanCaptureSE,
                                   queenCanCaptureSW, board.black, empty, collector};

    for (Bb x = board.whiteQueens() & queenCanCaptureNW; x; x &= x - 1) {
      Bb from = x & -x;
      gen.empty = empty | from;
      Bb cap = from;
      while (cap = moveNW(cap), !(cap & board.black));
      for (Bb to = moveNW(cap); to & empty; to = moveNW(to)) {
        if constexpr (Collector::tracks_path) {
          path = {std::countr_zero(from)};
        }
        gen.generate(from, to, cap, path);
      }
    }
    for (Bb x = board.whiteQueens() & queenCanCaptureNE; x; x &= x - 1) {
      Bb from = x & -x;
      gen.empty = empty | from;
      Bb cap = from;
      while (cap = moveNE(cap), !(cap & board.black));
      for (Bb to = moveNE(cap); to & empty; to = moveNE(to)) {
        if constexpr (Collector::tracks_path) {
          path = {std::countr_zero(from)};
        }
        gen.generate(from, to, cap, path);
      }
    }
    for (Bb x = board.whiteQueens() & queenCanCaptureSW; x; x &= x - 1) {
      Bb from = x & -x;
      gen.empty = empty | from;
      Bb cap = from;
      while (cap = moveSW(cap), !(cap & board.black));
      for (Bb to = moveSW(cap); to & empty; to = moveSW(to)) {
        if constexpr (Collector::tracks_path) {
          path = {std::countr_zero(from)};
        }
        gen.generate(from, to, cap, path);
      }
    }
    for (Bb x = board.whiteQueens() & queenCanCaptureSE; x; x &= x - 1) {
      Bb from = x & -x;
      gen.empty = empty | from;
      Bb cap = from;
      while (cap = moveSE(cap), !(cap & board.black));
      for (Bb to = moveSE(cap); to & empty; to = moveSE(to)) {
        if constexpr (Collector::tracks_path) {
          path = {std::countr_zero(from)};
        }
        gen.generate(from, to, cap, path);
      }
    }
  }
}

// Add quiet moves (templated)
template<typename Collector>
void addQuietMoves(const Board& board, Collector& collector) {
  Bb empty = board.empty();

  // Pawn quiet moves
  for (Bb x = board.whitePawns() & moveSE(empty); x; x &= x - 1) {
    Bb from = x & -x;
    Bb to = moveNW(from);
    if constexpr (Collector::tracks_path) {
      collector.add(Move(from ^ to, 0),
                    {std::countr_zero(from), std::countr_zero(to)});
    } else {
      collector.add(Move(from ^ to, 0));
    }
  }
  for (Bb x = board.whitePawns() & moveSW(empty); x; x &= x - 1) {
    Bb from = x & -x;
    Bb to = moveNE(from);
    if constexpr (Collector::tracks_path) {
      collector.add(Move(from ^ to, 0),
                    {std::countr_zero(from), std::countr_zero(to)});
    } else {
      collector.add(Move(from ^ to, 0));
    }
  }

  // Queen quiet moves
  for (Bb x = board.whiteQueens() & moveSE(empty); x; x &= x - 1) {
    Bb from = x & -x;
    for (Bb to = moveNW(from); to & empty; to = moveNW(to)) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, 0),
                      {std::countr_zero(from), std::countr_zero(to)});
      } else {
        collector.add(Move(from ^ to, 0));
      }
    }
  }
  for (Bb x = board.whiteQueens() & moveSW(empty); x; x &= x - 1) {
    Bb from = x & -x;
    for (Bb to = moveNE(from); to & empty; to = moveNE(to)) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, 0),
                      {std::countr_zero(from), std::countr_zero(to)});
      } else {
        collector.add(Move(from ^ to, 0));
      }
    }
  }
  for (Bb x = board.whiteQueens() & moveNW(empty); x; x &= x - 1) {
    Bb from = x & -x;
    for (Bb to = moveSE(from); to & empty; to = moveSE(to)) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, 0),
                      {std::countr_zero(from), std::countr_zero(to)});
      } else {
        collector.add(Move(from ^ to, 0));
      }
    }
  }
  for (Bb x = board.whiteQueens() & moveNE(empty); x; x &= x - 1) {
    Bb from = x & -x;
    for (Bb to = moveSW(from); to & empty; to = moveSW(to)) {
      if constexpr (Collector::tracks_path) {
        collector.add(Move(from ^ to, 0),
                      {std::countr_zero(from), std::countr_zero(to)});
      } else {
        collector.add(Move(from ^ to, 0));
      }
    }
  }
}

} // namespace

std::size_t generateMoves(const Board& board, MoveList& moves) {
  moves.clear();
  MoveCollector collector(moves);

  generateMovesImpl(board, collector);

  // Filter captures by ley de cantidad and deduplicate
  if (moves.size() > 1) {
    int maxCaptures = 0;
    for (const auto& m : moves)
      maxCaptures = std::max(maxCaptures, std::popcount(m.captures));

    // Filter: keep only moves with max captures
    std::size_t writeIdx = 0;
    for (std::size_t i = 0; i < moves.size(); ++i) {
      if (std::popcount(moves[i].captures) == maxCaptures) {
        moves[writeIdx++] = moves[i];
      }
    }
    moves.count = writeIdx;

    // Sort and deduplicate
    std::sort(moves.begin(), moves.end());
    std::size_t uniqueIdx = 0;
    for (std::size_t i = 0; i < moves.size(); ++i) {
      if (i == 0 || !(moves[i] == moves[uniqueIdx - 1])) {
        moves[uniqueIdx++] = moves[i];
      }
    }
    moves.count = uniqueIdx;
  }

  // Quiet moves (only if no captures)
  if (moves.empty()) {
    addQuietMoves(board, collector);
  }

  return moves.size();
}

uint64_t perft(const Board& board, int depth) {
  if (depth == 0) return 1;

  MoveList moves;
  generateMoves(board, moves);

  if (depth == 1) return moves.size();

  uint64_t nodes = 0;
  for (const auto& move : moves)
    nodes += perft(makeMove(board, move), depth - 1);

  return nodes;
}

std::size_t generateFullMoves(const Board& board, std::vector<FullMove>& moves) {
  moves.clear();
  FullMoveCollector collector(moves);

  generateMovesImpl(board, collector);

  // Filter captures by ley de cantidad
  if (moves.size() > 1) {
    int maxCaptures = 0;
    for (const auto& m : moves)
      maxCaptures = std::max(maxCaptures, std::popcount(m.move.captures));

    std::erase_if(moves, [maxCaptures](const FullMove& m) {
      return std::popcount(m.move.captures) < maxCaptures;
    });

    // Deduplicate by Move (keep first path for each unique move)
    std::vector<FullMove> unique;
    for (auto& m : moves) {
      bool found = false;
      for (const auto& u : unique) {
        if (u.move == m.move) {
          found = true;
          break;
        }
      }
      if (!found) {
        unique.push_back(std::move(m));
      }
    }
    moves = std::move(unique);
  }

  // Quiet moves (only if no captures)
  if (moves.empty()) {
    addQuietMoves(board, collector);
  }

  return moves.size();
}

bool has_captures(const Board& board) {
  Bb empty = board.empty();

  // Pawn captures
  Bb pawnCanCaptureNW = moveSE(board.black & moveSE(empty));
  Bb pawnCanCaptureNE = moveSW(board.black & moveSW(empty));
  if (board.whitePawns() & (pawnCanCaptureNW | pawnCanCaptureNE)) {
    return true;
  }

  // Queen captures
  Bb kingOrEmpty = board.whiteQueens() | empty;
  Bb canBeCapturedNW = board.black & moveSE(empty);
  Bb canBeCapturedNE = board.black & moveSW(empty);
  Bb canBeCapturedSE = board.black & moveNW(empty);
  Bb canBeCapturedSW = board.black & moveNE(empty);

  // Check if any queen can reach a capture position
  Bb queenCanCapture = 0;
  for (Bb x = moveSE(canBeCapturedNW) & kingOrEmpty; x; x = moveSE(x) & kingOrEmpty) {
    queenCanCapture |= x;
    x &= empty;
  }
  for (Bb x = moveSW(canBeCapturedNE) & kingOrEmpty; x; x = moveSW(x) & kingOrEmpty) {
    queenCanCapture |= x;
    x &= empty;
  }
  for (Bb x = moveNW(canBeCapturedSE) & kingOrEmpty; x; x = moveNW(x) & kingOrEmpty) {
    queenCanCapture |= x;
    x &= empty;
  }
  for (Bb x = moveNE(canBeCapturedSW) & kingOrEmpty; x; x = moveNE(x) & kingOrEmpty) {
    queenCanCapture |= x;
    x &= empty;
  }

  return (board.whiteQueens() & queenCanCapture) != 0;
}
