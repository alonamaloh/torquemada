#pragma once

#include "board.h"
#include <cstdint>
#include <vector>

// Move with full path information for notation display.
// Path contains the sequence of landing squares (0-31 indexed).
// For simple moves: [from, to]
// For captures: [from, land1, land2, ..., to] where each landN is where
// the piece lands after each capture.
struct FullMove {
  Move move;
  std::vector<int> path;

  FullMove() = default;
  FullMove(const Move& m, std::vector<int> p) : move(m), path(std::move(p)) {}
};

// Generate all legal moves for the side to move (white).
// Moves are added to the provided vector (cleared first).
// Returns the number of moves generated.
std::size_t generateMoves(const Board& board, std::vector<Move>& moves);

// Generate all legal moves with full path information for notation.
std::size_t generateFullMoves(const Board& board, std::vector<FullMove>& moves);

// Perft: count leaf nodes at given depth (for testing/debugging)
uint64_t perft(const Board& board, int depth);

// Check if the position has captures available for the side to move
bool has_captures(const Board& board);

