#pragma once

#include <cstdint>
#include <ostream>

using Bb = std::uint32_t;

// Bitboard direction functions - shift all bits in a direction at once
inline Bb moveNW(Bb x) { return ((x & 0x0f0f0f0fu) << 4) | ((x & 0x00707070u) << 5); }
inline Bb moveNE(Bb x) { return ((x & 0x0e0e0e0eu) << 3) | ((x & 0x00f0f0f0u) << 4); }
inline Bb moveSE(Bb x) { return ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0e0e0e00u) >> 5); }
inline Bb moveSW(Bb x) { return ((x & 0x70707070u) >> 3) | ((x & 0x0f0f0f00u) >> 4); }

// Flip a bitboard (rotate 180 degrees)
inline Bb flip(Bb x) {
  x = __builtin_bswap32(x);
  x = (x & 0xf0f0f0f0u) >> 4 | (x & 0x0f0f0f0fu) << 4;
  x = (x & 0xccccccccu) >> 2 | (x & 0x33333333u) << 2;
  x = (x & 0xaaaaaaaau) >> 1 | (x & 0x55555555u) << 1;
  return x;
}

// Compact move representation for search
struct Move {
  Bb from_xor_to;
  Bb captures;

  Move() : from_xor_to(0), captures(0) {}
  Move(Bb fxt, Bb cap) : from_xor_to(fxt), captures(cap) {}

  bool isCapture() const { return captures != 0; }

  bool operator==(const Move& other) const {
    return from_xor_to == other.from_xor_to && captures == other.captures;
  }
  bool operator<(const Move& other) const {
    if (from_xor_to != other.from_xor_to) return from_xor_to < other.from_xor_to;
    return captures < other.captures;
  }
};

// Fixed-size vector to avoid heap allocations in hot paths
template<typename T, int MaxSize>
struct FixedVector {
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  T data[MaxSize];
  int count = 0;

  T const& operator[](std::size_t i) const { return data[i]; }
  T& operator[](std::size_t i) { return data[i]; }

  void push_back(const T& val) { data[count++] = val; }

  void clear() { count = 0; }
  bool empty() const { return count == 0; }
  int size() const { return count; }

  iterator begin() { return data; }
  iterator end() { return data + count; }
  const_iterator begin() const { return data; }
  const_iterator end() const { return data + count; }
};

// Move list with fixed capacity (256 is an upper bound for legal moves)
using MoveList = FixedVector<Move, 256>;

// Board state - white is always the side to move
// After each move, the board is flipped so white remains to move
struct Board {
  Bb white;
  Bb black;
  Bb kings;
  unsigned n_reversible = 0;

  // Default: initial position
  Board(Bb w = 0x00000fffu, Bb b = 0xfff00000u, Bb k = 0)
      : white(w), black(b), kings(k) {}

  Bb whitePawns() const { return white & ~kings; }
  Bb whiteQueens() const { return white & kings; }
  Bb blackPawns() const { return black & ~kings; }
  Bb blackQueens() const { return black & kings; }
  Bb allPieces() const { return white | black; }
  Bb empty() const { return ~allPieces(); }

  // 64-bit hash using arithmetic mixing
  std::uint64_t hash() const {
    std::uint64_t h = white * 0x9d82c4a44a2de231ull;
    h ^= h >> 32;
    h += black;
    h *= 0xb20534a511d28c31ull;
    h ^= h >> 32;
    h += kings;
    h *= 0xb3fc4d1b1770e375ull;
    h ^= h >> 32;
    h += n_reversible;
    h *= 0x3a2a8392d61061d7ull;
    h ^= h >> 32;
    return h;
  }
};

// Allow Board to be used as a key in std::unordered_map/set
inline bool operator==(const Board& a, const Board& b) {
  return a.white == b.white && a.black == b.black &&
         a.kings == b.kings && a.n_reversible == b.n_reversible;
}

namespace std {
  template<>
  struct hash<Board> {
    std::size_t operator()(const Board& b) const noexcept {
      return b.hash();
    }
  };
}

// Flip the board (swap white/black, rotate 180 degrees)
inline Board flip(const Board& board) {
  Board b;
  b.white = flip(board.black);
  b.black = flip(board.white);
  b.kings = flip(board.kings);
  b.n_reversible = board.n_reversible;
  return b;
}

// Apply a move and flip the board (so white is always to move)
Board makeMove(Board board, const Move& move);

std::ostream& operator<<(std::ostream& os, const Board& board);

