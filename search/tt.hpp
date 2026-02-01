#pragma once

#include "../core/board.hpp"
#include <cstdint>

// Transposition table entry flag
enum class TTFlag : std::uint8_t {
  NONE = 0,
  EXACT = 1,      // Exact score (PV node)
  LOWER_BOUND = 2, // Score is a lower bound (beta cutoff)
  UPPER_BOUND = 3  // Score is an upper bound (failed low)
};

// Compact move representation for TT: from XOR to XOR captures
// This uniquely identifies a move and can be matched against generated moves
using CompactMove = std::uint32_t;

inline CompactMove move_to_compact(const Move& m) {
  return m.from_xor_to ^ m.captures;
}

inline bool compact_matches(CompactMove compact, const Move& m) {
  return compact == (m.from_xor_to ^ m.captures);
}

// Transposition table entry (16 bytes for cache efficiency)
// Using a 32-bit lock (upper bits of hash) instead of full 64-bit key for compactness
struct TTEntry {
  std::uint32_t lock;       // Upper 32 bits of hash for verification
  CompactMove best_move;    // Compact move representation (4 bytes)
  std::int16_t score;       // Evaluation score
  std::int8_t depth;        // Search depth
  TTFlag flag;              // Score type
  std::uint8_t generation;  // For replacement policy
  std::uint8_t padding[3];  // Pad to 16 bytes

  TTEntry() : lock(0), best_move(0), score(0), depth(0), flag(TTFlag::NONE), generation(0) {}
};

static_assert(sizeof(TTEntry) == 16, "TTEntry should be 16 bytes for cache efficiency");

// Transposition table with buckets of 4 entries
// Uses power-of-2 sizing for fast indexing
class TranspositionTable {
public:
  // Create table with approximately the given size in MB
  explicit TranspositionTable(std::size_t size_mb = 256);
  ~TranspositionTable();

  // Non-copyable
  TranspositionTable(const TranspositionTable&) = delete;
  TranspositionTable& operator=(const TranspositionTable&) = delete;

  // Movable
  TranspositionTable(TranspositionTable&& other) noexcept;
  TranspositionTable& operator=(TranspositionTable&& other) noexcept;

  // Clear all entries
  void clear();

  // Increment generation (call at start of each search)
  void new_search() { current_generation_++; }

  // Probe the table for a position
  // Returns true if a matching entry was found
  bool probe(std::uint64_t key, TTEntry& entry) const;

  // Store an entry in the table
  // Uses bucket replacement: prefer same hash, then lowest priority (depth - generation bonus)
  void store(std::uint64_t key, int score, int depth, TTFlag flag, const Move& best_move);

  // Get table statistics
  std::size_t size() const { return num_entries_; }
  std::size_t size_mb() const { return (num_entries_ * sizeof(TTEntry)) / (1024 * 1024); }

private:
  static constexpr int BUCKET_SIZE = 4;

  TTEntry* entries_ = nullptr;
  std::size_t num_entries_ = 0;
  std::size_t mask_ = 0;  // For fast modulo with power-of-2 size
  std::uint8_t current_generation_ = 0;
};
