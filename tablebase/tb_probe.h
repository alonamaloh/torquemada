#pragma once

#include "../core/board.h"
#include "tablebase.h"
#include "compression.h"
#include <string>
#include <memory>

namespace tablebase {

// Re-export the CompressedTablebaseManager for use by the search engine
using ::CompressedTablebaseManager;
using ::Value;
using ::Material;
using ::get_material;

// Convert tablebase Value to a search score
// WIN = side to move wins
// LOSS = side to move loses
// DRAW = draw
inline int value_to_score(Value v, int win_score, int loss_score, int draw_score) {
  switch (v) {
    case Value::WIN:  return win_score;
    case Value::LOSS: return loss_score;
    case Value::DRAW: return draw_score;
    default:          return 0;  // UNKNOWN
  }
}

// Check if a Value represents a known result
inline bool is_known(Value v) {
  return v != Value::UNKNOWN;
}

} // namespace tablebase
