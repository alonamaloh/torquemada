#pragma once

#include "../core/board.hpp"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <vector>
#include <string>

// Material configuration for tablebase indexing
// Pieces are divided into categories for gapless indexing:
// - back_white_pawns: White pawns on row 1 (squares 0-3), max 4
// - back_black_pawns: Black pawns on row 8 (squares 28-31), max 4
// - other_white_pawns: White pawns on rows 2-7 (squares 4-27), max 24
// - other_black_pawns: Black pawns on rows 2-7
// - white_queens/black_queens: Queens anywhere on remaining squares
struct Material {
  int back_white_pawns;
  int back_black_pawns;
  int other_white_pawns;
  int other_black_pawns;
  int white_queens;
  int black_queens;

  // Total piece count for each side
  int white_pieces() const { return back_white_pawns + other_white_pawns + white_queens; }
  int black_pieces() const { return back_black_pawns + other_black_pawns + black_queens; }
  int total_pieces() const { return white_pieces() + black_pieces(); }

  bool operator==(const Material& other) const {
    return back_white_pawns == other.back_white_pawns &&
           back_black_pawns == other.back_black_pawns &&
           other_white_pawns == other.other_white_pawns &&
           other_black_pawns == other.other_black_pawns &&
           white_queens == other.white_queens &&
           black_queens == other.black_queens;
  }
};

// Flip material (swap white and black)
inline Material flip(const Material& m) {
  return Material{
    m.back_black_pawns, m.back_white_pawns,
    m.other_black_pawns, m.other_white_pawns,
    m.black_queens, m.white_queens
  };
}

// String representation: "bbwwkK" format
inline std::ostream& operator<<(std::ostream& os, const Material& m) {
  return os << m.back_white_pawns << m.back_black_pawns
            << m.other_white_pawns << m.other_black_pawns
            << m.white_queens << m.black_queens;
}

// Hash for use in unordered containers
namespace std {
template<>
struct hash<Material> {
  std::size_t operator()(const Material& m) const noexcept {
    // Pack into single 64-bit value (each field < 32)
    std::uint64_t h = m.back_white_pawns;
    h = h * 33 + m.back_black_pawns;
    h = h * 33 + m.other_white_pawns;
    h = h * 33 + m.other_black_pawns;
    h = h * 33 + m.white_queens;
    h = h * 33 + m.black_queens;
    return h;
  }
};
}

// Tablebase value (2 bits per position)
enum class Value : std::uint8_t {
  UNKNOWN = 0,
  WIN = 1,      // Side to move wins
  LOSS = 2,     // Side to move loses
  DRAW = 3      // Draw (including positions with captures available)
};

// ============================================================================
// Indexing functions
// ============================================================================

// Extract material configuration from a board
Material get_material(const Board& b);

// Total number of positions for a material configuration
std::size_t material_size(const Material& m);

// Convert board to index (must have matching material)
std::size_t board_to_index(const Board& b, const Material& m);

// Convert index back to board
Board index_to_board(std::size_t idx, const Material& m);

// ============================================================================
// Tablebase storage
// ============================================================================

// Filename for a material configuration
std::string tablebase_filename(const Material& m);

// Filename for a compressed WDL tablebase
std::string compressed_tablebase_filename(const Material& m);

// Save tablebase to file
void save_tablebase(const std::vector<Value>& table, const Material& m);

// Load tablebase from file (returns empty vector if not found)
std::vector<Value> load_tablebase(const Material& m);

// Check if tablebase exists
bool tablebase_exists(const Material& m);

// ============================================================================
// DTM (Distance-To-Mate) storage
// ============================================================================

// DTM value: signed 16-bit integer, ordered by "happiness" (higher = better)
//
// Encoding (where M = number of moves for the winning side):
//   DTM =  M  : WIN, mate in M moves  (M = 1..127)
//   DTM =  0  : DRAW
//   DTM = -M  : LOSS, lost in M moves (M = 1..127)
//   DTM = -128: Terminal LOSS (no legal moves, game over)
//   DTM = -32768: UNKNOWN (not yet computed)
//
// "Happiness ordering": higher DTM = better outcome for side to move
//   - WIN in 1 move (DTM=1) > WIN in 2 moves (DTM=2) > ... (faster win = happier)
//   - Any WIN (>0) > DRAW (=0) > Any LOSS (<0)
//   - LOSS in 10 moves (DTM=-10) > LOSS in 1 move (DTM=-1) (survive longer = happier)
//
// For WIN positions: pick move leading to opponent's HIGHEST loss DTM
//   (least negative = quickest loss for opponent)
// For LOSS positions: pick move leading to opponent's HIGHEST win DTM
//   (most positive = slowest win for opponent, we survive longest)
//
// Plies vs Moves:
//   - A "move" is a turn by the winning side
//   - WIN in M moves = 2M-1 plies (we move M times, opponent moves M-1 times)
//   - LOSS in M moves = 2M plies (we move M times, opponent moves M times)
//
// Storage: lower byte (int8_t) sign-extends correctly to int16_t
using DTM = std::int16_t;

constexpr DTM DTM_DRAW = 0;
constexpr DTM DTM_LOSS_TERMINAL = -128;
constexpr DTM DTM_UNKNOWN = std::numeric_limits<std::int16_t>::min();  // -32768

// Encoding helpers (moves, not plies)
inline DTM dtm_win(int moves) { return static_cast<DTM>(moves); }
inline DTM dtm_loss(int moves) { return moves == 0 ? DTM_LOSS_TERMINAL : static_cast<DTM>(-moves); }

// Convert to/from plies (for display)
inline int dtm_to_plies(DTM d) {
  if (d > 0) return 2 * d - 1;   // WIN in M moves = 2M-1 plies
  if (d == DTM_LOSS_TERMINAL) return 0;  // Terminal = 0 plies
  if (d < 0 && d != DTM_UNKNOWN) return -2 * d;  // LOSS in M moves = 2M plies
  return 0;  // DRAW
}
inline int dtm_to_moves(DTM d) {
  if (d > 0) return d;
  if (d == DTM_LOSS_TERMINAL) return 0;
  if (d < 0 && d != DTM_UNKNOWN) return -d;
  return 0;
}

// DTM file format version
constexpr std::uint8_t DTM_FORMAT_VERSION = 1;  // 1-byte encoding

// Filename for DTM tablebase
std::string dtm_filename(const Material& m);

// Save DTM tablebase to file
void save_dtm(const std::vector<DTM>& table, const Material& m);

// Load DTM tablebase from file (returns empty vector if not found)
std::vector<DTM> load_dtm(const Material& m);

// Check if DTM tablebase exists
bool dtm_exists(const Material& m);

// ============================================================================
// Combinatorics helper
// ============================================================================

// Binomial coefficient C(n, k)
std::size_t choose(int n, int k);
