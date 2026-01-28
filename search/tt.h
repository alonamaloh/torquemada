#pragma once

#include "../core/board.h"
#include <cstdint>
#include <vector>

// Transposition table entry flag
enum class TTFlag : std::uint8_t {
  NONE = 0,
  EXACT = 1,      // Exact score (PV node)
  LOWER_BOUND = 2, // Score is a lower bound (beta cutoff)
  UPPER_BOUND = 3  // Score is an upper bound (failed low)
};

// Transposition table entry (24 bytes with alignment)
struct TTEntry {
  std::uint64_t key;      // Position hash (full 64 bits for verification)
  std::int16_t score;     // Evaluation score
  std::int8_t depth;      // Search depth
  TTFlag flag;            // Score type
  Move best_move;         // Best move found (8 bytes)

  TTEntry() : key(0), score(0), depth(0), flag(TTFlag::NONE), best_move() {}
};

// Transposition table with power-of-2 sizing for fast indexing
class TranspositionTable {
public:
  // Create table with approximately the given size in MB
  explicit TranspositionTable(std::size_t size_mb = 64);

  // Clear all entries
  void clear();

  // Probe the table for a position
  // Returns true if a matching entry was found
  bool probe(std::uint64_t key, TTEntry& entry) const;

  // Store an entry in the table
  // Uses replace-always strategy for simplicity
  void store(std::uint64_t key, int score, int depth, TTFlag flag, const Move& best_move);

  // Get table statistics
  std::size_t size() const { return entries_.size(); }
  std::size_t size_mb() const { return (entries_.size() * sizeof(TTEntry)) / (1024 * 1024); }

private:
  std::vector<TTEntry> entries_;
  std::size_t mask_;  // For fast modulo with power-of-2 size
};
