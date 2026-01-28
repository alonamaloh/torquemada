#include "tt.h"
#include <bit>

TranspositionTable::TranspositionTable(std::size_t size_mb) {
  // Calculate number of entries
  std::size_t bytes = size_mb * 1024 * 1024;
  std::size_t num_entries = bytes / sizeof(TTEntry);

  // Round down to power of 2 for fast indexing
  if (num_entries > 0) {
    num_entries = std::size_t{1} << (63 - std::countl_zero(num_entries));
  } else {
    num_entries = 1024;  // Minimum size
  }

  entries_.resize(num_entries);
  mask_ = num_entries - 1;
  clear();
}

void TranspositionTable::clear() {
  for (auto& entry : entries_) {
    entry = TTEntry{};
  }
}

bool TranspositionTable::probe(std::uint64_t key, TTEntry& entry) const {
  std::size_t index = key & mask_;
  const TTEntry& stored = entries_[index];

  if (stored.key == key && stored.flag != TTFlag::NONE) {
    entry = stored;
    return true;
  }
  return false;
}

void TranspositionTable::store(std::uint64_t key, int score, int depth, TTFlag flag,
                                const Move& best_move) {
  std::size_t index = key & mask_;
  TTEntry& entry = entries_[index];

  // Replace-always strategy (could be improved with age/depth considerations)
  entry.key = key;
  entry.score = static_cast<std::int16_t>(score);
  entry.depth = static_cast<std::int8_t>(depth);
  entry.flag = flag;
  entry.best_move = best_move;
}
