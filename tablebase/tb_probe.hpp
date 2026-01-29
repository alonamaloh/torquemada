#pragma once

#include "../core/board.hpp"
#include "../core/movegen.hpp"
#include "tablebase.hpp"
#include "compression.hpp"
#include <string>
#include <memory>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tablebase {

// Re-export types for use by the search engine
using ::CompressedTablebaseManager;
using ::Value;
using ::Material;
using ::DTM;
using ::DTM_DRAW;
using ::DTM_UNKNOWN;
using ::DTM_LOSS_TERMINAL;
using ::get_material;
using ::board_to_index;
using ::material_size;
using ::load_dtm;
using ::dtm_exists;

// Convert tablebase Value to a search score
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

// DTM Tablebase Manager - loads and caches DTM tablebases
class DTMTablebaseManager {
public:
  explicit DTMTablebaseManager(const std::string& directory)
      : directory_(directory) {}

  // Lookup DTM value for a position
  // Returns DTM_UNKNOWN if tablebase not available
  DTM lookup_dtm(const Board& board) {
    Material m = get_material(board);

    // Try to get or load the tablebase
    auto it = dtm_cache_.find(m);
    if (it == dtm_cache_.end()) {
      // Try to load
      if (!dtm_exists_in_dir(m)) {
        // Cache empty vector to avoid repeated filesystem checks
        dtm_cache_[m] = {};
        return DTM_UNKNOWN;
      }
      dtm_cache_[m] = load_dtm_from_dir(m);
      it = dtm_cache_.find(m);
    }

    if (it->second.empty()) {
      return DTM_UNKNOWN;
    }

    std::size_t idx = board_to_index(board, m);
    if (idx >= it->second.size()) {
      return DTM_UNKNOWN;
    }

    return it->second[idx];
  }

  // Check if DTM tablebase is available for a position
  bool has_dtm(const Board& board) {
    Material m = get_material(board);
    auto it = dtm_cache_.find(m);
    if (it != dtm_cache_.end()) {
      return !it->second.empty();
    }
    return dtm_exists(m);
  }

  // Find best move using DTM tablebase (depth-1 search with DTM lookup)
  // Logic adapted from damas/play_optimal.cpp
  // Returns true if a move was found, sets best_move and best_dtm (from our perspective)
  bool find_best_move(const Board& board, Move& best_move, DTM& best_dtm) {
    std::vector<Move> moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      return false;
    }

    // First, get our current DTM to know if we're winning/losing/drawing
    DTM current_dtm = lookup_dtm(board);

    DTM best_opp_dtm = DTM_UNKNOWN;
    bool found = false;

    for (const Move& move : moves) {
      Board child = makeMove(board, move);

      // Check if opponent has no pieces (we captured their last piece)
      Material child_m = get_material(child);
      DTM opp_dtm;
      if (child_m.white_pieces() == 0) {
        opp_dtm = DTM_LOSS_TERMINAL;  // Opponent has no pieces = terminal loss for them
      } else {
        opp_dtm = lookup_dtm(child);
      }

      if (opp_dtm == DTM_UNKNOWN) {
        continue;  // Can't evaluate this move
      }

      // Determine if this move is better than our current best
      // Logic from damas/play_optimal.cpp
      bool dominated = false;
      if (!found) {
        dominated = false;  // First valid move
      } else if (current_dtm > 0) {
        // We're winning - want opponent to lose as fast as possible
        if (best_opp_dtm < 0 && opp_dtm >= 0) {
          dominated = true;  // Current best loses them, new doesn't
        } else if (best_opp_dtm >= 0 && opp_dtm < 0) {
          dominated = false;  // New loses them, current doesn't
        } else if (best_opp_dtm < 0 && opp_dtm < 0) {
          // Both lose them - prefer faster loss (least negative = closer to 0)
          dominated = (opp_dtm <= best_opp_dtm);
        } else {
          dominated = true;  // Neither loses them (shouldn't happen)
        }
      } else if (current_dtm < 0) {
        // We're losing - want to survive as long as possible
        // Prefer higher opponent DTM (they take longer to win)
        dominated = (opp_dtm <= best_opp_dtm);
      } else {
        // Draw - prefer to keep it a draw or find a win
        dominated = (opp_dtm <= best_opp_dtm);
      }

      if (!dominated) {
        best_opp_dtm = opp_dtm;
        best_move = move;
        found = true;
      }
    }

    if (found) {
      // Convert opponent's DTM to our DTM
      if (best_opp_dtm == DTM_LOSS_TERMINAL) {
        best_dtm = 1;  // We win in 1 move
      } else if (best_opp_dtm < 0) {
        // Opponent loses in -best_opp_dtm moves, so we win
        best_dtm = static_cast<DTM>(-best_opp_dtm);
      } else if (best_opp_dtm == 0) {
        best_dtm = 0;  // Draw
      } else {
        // Opponent wins in best_opp_dtm moves, so we lose
        best_dtm = static_cast<DTM>(-best_opp_dtm);
      }
    }

    return found;
  }

  void clear() { dtm_cache_.clear(); }

private:
  std::string directory_;
  std::unordered_map<Material, std::vector<DTM>> dtm_cache_;

  // Helper to build full DTM filename with directory
  std::string dtm_path(const Material& m) const {
    std::ostringstream oss;
    oss << directory_ << "/dtm_"
        << m.back_white_pawns << m.back_black_pawns
        << m.other_white_pawns << m.other_black_pawns
        << m.white_queens << m.black_queens << ".bin";
    return oss.str();
  }

  // Check if DTM file exists in our directory
  bool dtm_exists_in_dir(const Material& m) const {
    return std::filesystem::exists(dtm_path(m));
  }

  // Load DTM from our directory
  // Throws std::runtime_error if file exists but cannot be properly read
  std::vector<DTM> load_dtm_from_dir(const Material& m) const {
    std::string path = dtm_path(m);
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open DTM file: " + path);
    }

    // Read format version
    std::uint8_t version;
    file.read(reinterpret_cast<char*>(&version), 1);
    if (!file) {
      throw std::runtime_error("Failed to read version from DTM file: " + path);
    }
    if (version != 1) {
      throw std::runtime_error("Unsupported DTM file version " + std::to_string(version) + " in: " + path);
    }

    // Read and verify material (sizeof(Material) = 24 bytes with int fields)
    Material stored_m;
    file.read(reinterpret_cast<char*>(&stored_m), sizeof(Material));
    if (!file) {
      throw std::runtime_error("Failed to read material from DTM file: " + path);
    }
    if (!(stored_m == m)) {
      throw std::runtime_error("Material mismatch in DTM file: " + path);
    }

    // Read number of positions
    std::size_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!file) {
      throw std::runtime_error("Failed to read count from DTM file: " + path);
    }

    // Read packed DTM values (1 byte each, signed)
    std::vector<std::int8_t> packed(count);
    file.read(reinterpret_cast<char*>(packed.data()), count);
    if (!file) {
      throw std::runtime_error("Failed to read data from DTM file: " + path);
    }

    // Convert to DTM
    std::vector<DTM> table(count);
    for (std::size_t i = 0; i < count; ++i) {
      table[i] = static_cast<DTM>(packed[i]);
    }

    return table;
  }
};

} // namespace tablebase
