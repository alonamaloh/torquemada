#include "board.h"
#include <bit>

// Square numbering (32 light squares, standard Spanish checkers convention):
//
//   31  30  29  28      (row 8 - black's back rank)
//     27  26  25  24    (row 7)
//   23  22  21  20      (row 6)
//     19  18  17  16    (row 5)
//   15  14  13  12      (row 4)
//     11  10  09  08    (row 3)
//   07  06  05  04      (row 2)
//     03  02  01  00    (row 1 - white's back rank)
//
// Note: Internally we use 0-31 for bitboard indexing. In standard notation
// (e.g., when writing moves as strings), squares are numbered 1-32.

Board makeMove(Board board, const Move& move) {
  // Update reversible moves counter
  if ((board.kings & move.from_xor_to) && move.captures == 0)
    ++board.n_reversible;
  else
    board.n_reversible = 0;

  // Move king if applicable
  if (board.kings & move.from_xor_to)
    board.kings ^= move.from_xor_to;

  // Move the white piece
  board.white ^= move.from_xor_to;

  // Remove captured pieces
  board.black &= ~move.captures;
  board.kings &= ~move.captures;

  // Promote pawns reaching the back rank
  board.kings |= board.white & 0xf0000000u;

  // Flip so white is always to move
  return flip(board);
}

std::ostream& operator<<(std::ostream& os, const Board& board) {
  for (int row = 7; row >= 0; --row) {
    if (row % 2 == 0) os << "  ";
    for (int col = 3; col >= 0; --col) {
      int bit = row * 4 + col;
      int code = ((board.white >> bit) & 1) +
                 ((board.black >> bit) & 1) * 2 +
                 ((board.kings >> bit) & 1) * 4;
      os << "_ox**OX*"[code] << "   ";
    }
    os << '\n';
  }
  os << "Reversible moves: " << board.n_reversible << '\n';
  return os;
}

