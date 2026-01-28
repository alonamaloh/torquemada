#include "compression.h"
#include "../core/movegen.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <omp.h>
#include <unordered_set>

// Forward declarations for optimized 2-value RLE
std::vector<std::uint8_t> compress_rle_huffman_2val(const Value* values, std::size_t count);
std::vector<Value> decompress_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values);

// Forward declarations for optimized 3-value RLE with prediction
std::vector<std::uint8_t> compress_rle_huffman_3val(const Value* values, std::size_t count);
std::vector<Value> decompress_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values);

// Forward declarations for on-the-fly Huffman lookups
static Value lookup_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                                      std::size_t target_idx);
static Value lookup_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                                      std::size_t target_idx);

// ============================================================================
// Don't-Care Position Detection (Stage 1)
// ============================================================================

bool is_dont_care(const Board& b) {
  // Only mark positions where WE have captures as don't-care.
  // These are truly forced (mandatory capture rule) and lead to material change.
  //
  // Note: The original plan also included "opponent would have captures" positions,
  // but these can lead to very long search chains through quiet moves, causing
  // performance issues. A future optimization could handle these differently
  // (e.g., limited depth search or separate treatment).
  return has_captures(b);
}

// ============================================================================
// Compression Statistics Analysis
// ============================================================================

CompressionStats analyze_compression(const std::vector<Value>& tablebase, const Material& m) {
  CompressionStats stats;
  stats.total_positions = tablebase.size();

  for (std::size_t idx = 0; idx < tablebase.size(); ++idx) {
    Board board = index_to_board(idx, m);
    Value val = tablebase[idx];

    // Check if this is a don't-care position (only "we have captures")
    bool we_have_captures = has_captures(board);

    if (we_have_captures) {
      stats.dont_care_positions++;
    } else {
      stats.real_positions++;
    }

    // Also track opponent captures for analysis (not used for don't-care)
    if (!we_have_captures && has_captures(flip(board))) {
      stats.opponent_capture_positions++;
    }

    // Count values
    switch (val) {
      case Value::WIN: stats.wins++; break;
      case Value::LOSS: stats.losses++; break;
      case Value::DRAW: stats.draws++; break;
      default: break;
    }
  }

  return stats;
}

// ============================================================================
// Mark Don't-Care Positions
// ============================================================================

std::vector<Value> mark_dont_care_positions(
    const std::vector<Value>& original,
    const Material& m,
    CompressionStats& stats) {

  stats = CompressionStats{};
  stats.total_positions = original.size();

  std::vector<Value> result(original.size());

  // Use reduction for statistics counters
  std::size_t dont_care = 0, real = 0, opp_capture = 0;
  std::size_t wins = 0, losses = 0, draws = 0;

  #pragma omp parallel for reduction(+:dont_care,real,opp_capture,wins,losses,draws)
  for (std::size_t idx = 0; idx < original.size(); ++idx) {
    Board board = index_to_board(idx, m);
    Value val = original[idx];

    // Check if this is a don't-care position (only "we have captures")
    bool we_have_captures = has_captures(board);

    if (we_have_captures) {
      dont_care++;
      result[idx] = Value::UNKNOWN;  // Use UNKNOWN as sentinel for don't-care
    } else {
      real++;
      result[idx] = val;
    }

    // Also track opponent captures for analysis (not used for don't-care)
    if (!we_have_captures && has_captures(flip(board))) {
      opp_capture++;
    }

    // Count original values
    switch (val) {
      case Value::WIN: wins++; break;
      case Value::LOSS: losses++; break;
      case Value::DRAW: draws++; break;
      default: break;
    }
  }

  stats.dont_care_positions = dont_care;
  stats.real_positions = real;
  stats.opponent_capture_positions = opp_capture;
  stats.wins = wins;
  stats.losses = losses;
  stats.draws = draws;

  return result;
}

// ============================================================================
// WDL Lookup with Search (Stage 2)
// ============================================================================

// Score encoding: WIN=1, DRAW=0, LOSS=-1
// This allows standard negamax: score = -negamax(child)
static constexpr int SCORE_WIN = 1;
static constexpr int SCORE_DRAW = 0;
static constexpr int SCORE_LOSS = -1;

static inline int value_to_score(Value v) {
  switch (v) {
    case Value::WIN: return SCORE_WIN;
    case Value::DRAW: return SCORE_DRAW;
    case Value::LOSS: return SCORE_LOSS;
    default: return SCORE_DRAW;  // Treat UNKNOWN conservatively as draw
  }
}

static inline Value score_to_value(int score) {
  if (score > 0) return Value::WIN;
  if (score < 0) return Value::LOSS;
  return Value::DRAW;
}

// Negamax search with alpha-beta pruning through don't-care positions.
// The Lookup callable returns an int score for quiet positions.
// Template to allow different lookup strategies (uncompressed, compressed, manager).
// Max depth is bounded by piece count (captures remove pieces), but we add a
// safety limit to prevent stack overflow from any bugs.
constexpr int MAX_NEGAMAX_DEPTH = 32;

template<typename Lookup>
static int negamax(const Board& b, int alpha, int beta, Lookup&& lookup,
                   SearchStats* stats = nullptr, int depth = 0) {
  // Quiet position: use the lookup function
  if (!has_captures(b)) {
    return lookup(b);
  }

  // Safety limit to prevent stack overflow - this should never trigger
  // with valid positions (max captures = total pieces - 1)
  if (depth >= MAX_NEGAMAX_DEPTH) {
    std::cerr << "\nFATAL: negamax depth " << depth << " exceeded limit!\n"
              << "Board: " << b << "\n"
              << "Material: " << get_material(b) << "\n"
              << "has_captures=" << has_captures(b) << "\n";
    std::vector<Move> dbg_moves;
    generateMoves(b, dbg_moves);
    std::cerr << "Generated " << dbg_moves.size() << " moves:\n";
    for (const auto& mv : dbg_moves) {
      Board next = makeMove(b, mv);
      std::cerr << "  Move from_xor_to=0x" << std::hex << mv.from_xor_to
                << " captures=0x" << mv.captures << std::dec
                << " -> Material: " << get_material(next) << "\n";
    }
    std::exit(1);
  }

  // Capture position: search through forced moves
  std::vector<Move> moves;
  generateMoves(b, moves);

  if (moves.empty()) {
    return SCORE_LOSS;
  }

  for (const Move& move : moves) {
    Board next = makeMove(b, move);

    // Terminal: captured all opponent pieces
    if (next.white == 0) {
      if (stats) stats->terminal_wins++;
      return SCORE_WIN;
    }

    int score = -negamax(next, -beta, -alpha, lookup, stats, depth + 1);

    if (score >= beta) {
      return score;  // Beta cutoff
    }
    alpha = std::max(alpha, score);
  }

  return alpha;
}

Value lookup_wdl_with_search(
    const Board& b,
    const std::vector<Value>& tablebase,
    const Material& m) {

  auto lookup = [&](const Board& pos) -> int {
    Material pos_m = get_material(pos);
    if (!(pos_m == m)) {
      return SCORE_DRAW;  // Material changed, no sub-tablebase - conservative
    }
    std::size_t idx = board_to_index(pos, m);
    return value_to_score(tablebase[idx]);
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}

Value lookup_wdl_with_search(
    const Board& b,
    const std::vector<Value>& tablebase,
    const Material& m,
    const std::unordered_map<Material, std::vector<Value>>& sub_tablebases,
    SearchStats* stats) {

  if (stats) stats->lookups++;

  // Check if it's a direct hit (not don't-care)
  if (!has_captures(b)) {
    std::size_t idx = board_to_index(b, m);
    Value stored = tablebase[idx];
    if (stats) stats->direct_hits++;
    return stored;
  }

  // Need to search
  if (stats) stats->searches++;

  auto lookup = [&](const Board& pos) -> int {
    if (stats) stats->total_nodes++;

    Material pos_m = get_material(pos);

    // Material changed - look up in sub-tablebase
    if (!(pos_m == m)) {
      if (stats) stats->sub_tb_lookups++;
      auto it = sub_tablebases.find(pos_m);
      if (it == sub_tablebases.end() || it->second.empty()) {
        return SCORE_DRAW;  // Conservative
      }
      // Recurse into sub-tablebase
      std::size_t idx = board_to_index(pos, pos_m);
      return value_to_score(it->second[idx]);
    }

    std::size_t idx = board_to_index(pos, m);
    return value_to_score(tablebase[idx]);
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup, stats);
  return score_to_value(score);
}

// ============================================================================
// Block Compression/Decompression (Stage 3)
// ============================================================================

namespace {

// Value encoding for compression:
// Map Value enum to small integers for compact storage.
// We use: WIN=0, DRAW=1, LOSS=2, UNKNOWN=3
inline std::uint8_t value_to_int(Value v) {
  switch (v) {
    case Value::WIN: return 0;
    case Value::DRAW: return 1;
    case Value::LOSS: return 2;
    case Value::UNKNOWN: return 3;
    default: return 3;  // Treat invalid as UNKNOWN
  }
}

inline Value int_to_value(std::uint8_t i) {
  switch (i) {
    case 0: return Value::WIN;
    case 1: return Value::DRAW;
    case 2: return Value::LOSS;
    case 3: return Value::UNKNOWN;  // Don't-care position
    default:
      std::cerr << "int_to_value: invalid value " << static_cast<int>(i) << "\n";
      std::exit(1);
  }
}

// ============================================================================
// Method 0: RAW_2BIT
// 2 bits per value, 4 values per byte
// ============================================================================

std::vector<std::uint8_t> compress_raw_2bit(const Value* values, std::size_t count) {
  std::size_t num_bytes = (count + 3) / 4;
  std::vector<std::uint8_t> result(num_bytes, 0);

  for (std::size_t i = 0; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    std::size_t byte_idx = i / 4;
    std::size_t bit_pos = (i % 4) * 2;
    result[byte_idx] |= (v << bit_pos);
  }

  return result;
}

std::vector<Value> decompress_raw_2bit(const std::uint8_t* data, std::size_t num_values) {
  std::vector<Value> result(num_values);

  for (std::size_t i = 0; i < num_values; ++i) {
    std::size_t byte_idx = i / 4;
    std::size_t bit_pos = (i % 4) * 2;
    std::uint8_t v = (data[byte_idx] >> bit_pos) & 0x3;
    result[i] = int_to_value(v);
  }

  return result;
}

// ============================================================================
// Method 3: RLE_BINARY_SEARCH
// Run-length encoding with 16-bit records: 14-bit index + 2-bit value.
// Each record means "from this index until the next record, value is X".
// Lookup is O(log n) via binary search on indices.
// ============================================================================

// Compress using RLE with 16-bit records.
// Format: sequence of 16-bit little-endian records
//   Bits 0-13: start index (0-16383)
//   Bits 14-15: value (0-3)
// Records are sorted by index. The last record covers to end of block.
std::vector<std::uint8_t> compress_rle_binary_search(const Value* values, std::size_t count) {
  if (count == 0) return {};

  std::vector<std::uint8_t> result;
  result.reserve(count / 100 * 2);  // Estimate: ~1% run changes

  std::uint8_t current_value = value_to_int(values[0]);

  // First record always starts at index 0
  std::uint16_t record = (static_cast<std::uint16_t>(current_value) << 14) | 0;
  result.push_back(record & 0xFF);
  result.push_back((record >> 8) & 0xFF);

  for (std::size_t i = 1; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (v != current_value) {
      // New run starts at index i
      current_value = v;
      record = (static_cast<std::uint16_t>(v) << 14) | static_cast<std::uint16_t>(i);
      result.push_back(record & 0xFF);
      result.push_back((record >> 8) & 0xFF);
    }
  }

  return result;
}

std::vector<Value> decompress_rle_binary_search(const std::uint8_t* data, std::size_t data_size, std::size_t num_values) {
  if (num_values == 0 || data_size < 2) return std::vector<Value>(num_values, Value::UNKNOWN);

  std::vector<Value> result(num_values);
  std::size_t num_records = data_size / 2;

  // Decode all records
  std::vector<std::pair<std::size_t, Value>> runs;
  runs.reserve(num_records);

  for (std::size_t i = 0; i < num_records; ++i) {
    std::uint16_t record = data[i * 2] | (data[i * 2 + 1] << 8);
    std::size_t start_idx = record & 0x3FFF;
    std::uint8_t val = (record >> 14) & 0x3;
    runs.emplace_back(start_idx, int_to_value(val));
  }

  // Fill result by iterating through runs
  for (std::size_t r = 0; r < runs.size(); ++r) {
    std::size_t start = runs[r].first;
    std::size_t end = (r + 1 < runs.size()) ? runs[r + 1].first : num_values;
    Value val = runs[r].second;

    for (std::size_t i = start; i < end; ++i) {
      result[i] = val;
    }
  }

  return result;
}

// Lookup a single value using binary search on RLE records.
Value lookup_rle_binary_search(const std::uint8_t* data, std::size_t data_size, std::size_t index) {
  std::size_t num_records = data_size / 2;

  // Binary search to find the record that covers this index.
  // We want the largest start_idx <= index.
  std::size_t lo = 0, hi = num_records;

  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    std::uint16_t record = data[mid * 2] | (data[mid * 2 + 1] << 8);
    std::size_t start_idx = record & 0x3FFF;

    if (start_idx <= index) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // lo is now the first record with start_idx > index, so lo-1 is our record
  std::uint16_t record = data[(lo - 1) * 2] | (data[(lo - 1) * 2 + 1] << 8);
  std::uint8_t val = (record >> 14) & 0x3;
  return int_to_value(val);
}

// ============================================================================
// Method 2: DEFAULT_EXCEPTIONS
// Stores a default value plus a sorted list of exceptions.
// Format:
//   Byte 0: Default value (2 bits in low bits)
//   Bytes 1-2: Number of exceptions (uint16_t, little-endian)
//   Bytes 3+: Exceptions, each 2 bytes (14-bit index + 2-bit value)
// ============================================================================

std::vector<std::uint8_t> compress_default_exceptions(const Value* values, std::size_t count) {
  if (count == 0) return {0, 0, 0};  // Default=0, 0 exceptions

  // Count occurrences of each value to find the most common (default)
  std::size_t value_counts[4] = {0, 0, 0, 0};
  for (std::size_t i = 0; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    value_counts[v]++;
  }

  // Find the most common value
  std::uint8_t default_value = 0;
  std::size_t max_count = value_counts[0];
  for (std::uint8_t v = 1; v < 4; ++v) {
    if (value_counts[v] > max_count) {
      max_count = value_counts[v];
      default_value = v;
    }
  }

  // Collect exceptions (positions where value != default)
  std::vector<std::uint16_t> exceptions;
  exceptions.reserve(count - max_count);

  for (std::size_t i = 0; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (v != default_value) {
      // Pack: 14-bit index (low bits) + 2-bit value (high bits)
      std::uint16_t entry = (static_cast<std::uint16_t>(v) << 14) | static_cast<std::uint16_t>(i);
      exceptions.push_back(entry);
    }
  }

  // Build result: header (3 bytes) + exceptions (2 bytes each)
  std::vector<std::uint8_t> result;
  result.reserve(3 + exceptions.size() * 2);

  // Byte 0: default value
  result.push_back(default_value);

  // Bytes 1-2: exception count
  std::uint16_t num_exceptions = static_cast<std::uint16_t>(exceptions.size());
  result.push_back(num_exceptions & 0xFF);
  result.push_back((num_exceptions >> 8) & 0xFF);

  // Exceptions (already sorted by index since we iterate in order)
  for (std::uint16_t entry : exceptions) {
    result.push_back(entry & 0xFF);
    result.push_back((entry >> 8) & 0xFF);
  }

  return result;
}

std::vector<Value> decompress_default_exceptions(const std::uint8_t* data, std::size_t data_size, std::size_t num_values) {
  if (data_size < 3) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Read header
  std::uint8_t default_value = data[0] & 0x3;
  std::uint16_t num_exceptions = data[1] | (data[2] << 8);

  // Initialize all positions to default value
  std::vector<Value> result(num_values, int_to_value(default_value));

  // Apply exceptions
  for (std::size_t i = 0; i < num_exceptions && (3 + i * 2 + 1) < data_size; ++i) {
    std::uint16_t entry = data[3 + i * 2] | (data[3 + i * 2 + 1] << 8);
    std::size_t idx = entry & 0x3FFF;
    std::uint8_t val = (entry >> 14) & 0x3;

    if (idx < num_values) {
      result[idx] = int_to_value(val);
    }
  }

  return result;
}

// Lookup a single value using binary search on exception list.
Value lookup_default_exceptions(const std::uint8_t* data, std::size_t /*data_size*/, std::size_t index) {
  // Read header
  std::uint8_t default_value = data[0] & 0x3;
  std::uint16_t num_exceptions = data[1] | (data[2] << 8);

  if (num_exceptions == 0) {
    return int_to_value(default_value);
  }

  // Binary search for this index in the exception list
  const std::uint8_t* exceptions = data + 3;
  std::size_t lo = 0, hi = num_exceptions;

  while (lo < hi) {
    std::size_t mid = lo + (hi - lo) / 2;
    std::uint16_t entry = exceptions[mid * 2] | (exceptions[mid * 2 + 1] << 8);
    std::size_t exc_idx = entry & 0x3FFF;

    if (exc_idx == index) {
      // Found it - return the exception value
      std::uint8_t val = (entry >> 14) & 0x3;
      return int_to_value(val);
    } else if (exc_idx < index) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Not found in exceptions - return default value
  return int_to_value(default_value);
}

} // anonymous namespace

std::vector<std::uint8_t> compress_block(
    const Value* values,
    std::size_t count,
    CompressionMethod method) {

  switch (method) {
    case CompressionMethod::RAW_2BIT:
      return compress_raw_2bit(values, count);

    case CompressionMethod::RLE_BINARY_SEARCH:
      return compress_rle_binary_search(values, count);

    case CompressionMethod::DEFAULT_EXCEPTIONS:
      return compress_default_exceptions(values, count);

    case CompressionMethod::RLE_HUFFMAN_2VAL:
      return compress_rle_huffman_2val(values, count);

    case CompressionMethod::RLE_HUFFMAN_3VAL:
      return compress_rle_huffman_3val(values, count);

    default:
      return compress_raw_2bit(values, count);
  }
}

std::vector<Value> decompress_block(
    const std::uint8_t* data,
    std::size_t data_size,
    std::size_t num_values,
    CompressionMethod method) {

  switch (method) {
    case CompressionMethod::RAW_2BIT:
      return decompress_raw_2bit(data, num_values);

    case CompressionMethod::RLE_BINARY_SEARCH:
      return decompress_rle_binary_search(data, data_size, num_values);

    case CompressionMethod::DEFAULT_EXCEPTIONS:
      return decompress_default_exceptions(data, data_size, num_values);

    case CompressionMethod::RLE_HUFFMAN_2VAL:
      return decompress_rle_huffman_2val(data, data_size, num_values);

    case CompressionMethod::RLE_HUFFMAN_3VAL:
      return decompress_rle_huffman_3val(data, data_size, num_values);

    default:
      return decompress_raw_2bit(data, num_values);
  }
}

std::pair<CompressionMethod, std::vector<std::uint8_t>> compress_block_best(
    const Value* values,
    std::size_t count) {

  // Try Method 0: RAW_2BIT (always available)
  auto raw = compress_raw_2bit(values, count);
  CompressionMethod best_method = CompressionMethod::RAW_2BIT;
  std::vector<std::uint8_t> best_data = std::move(raw);

  // Try Method 3: RLE_BINARY_SEARCH (always available)
  auto rle = compress_rle_binary_search(values, count);
  if (rle.size() < best_data.size()) {
    best_method = CompressionMethod::RLE_BINARY_SEARCH;
    best_data = std::move(rle);
  }

  // Try Method 2: DEFAULT_EXCEPTIONS (always available)
  auto def_exc = compress_default_exceptions(values, count);
  if (def_exc.size() < best_data.size()) {
    best_method = CompressionMethod::DEFAULT_EXCEPTIONS;
    best_data = std::move(def_exc);
  }

  // Count distinct values to determine which methods to try
  bool seen[4] = {false, false, false, false};
  int num_distinct = 0;
  for (std::size_t i = 0; i < count && num_distinct <= 3; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (!seen[v]) {
      seen[v] = true;
      num_distinct++;
    }
  }

  // Try Method 8: RLE_HUFFMAN_2VAL (optimized for 2-value blocks)
  if (num_distinct == 2) {
    auto huffman_2val = compress_rle_huffman_2val(values, count);
    if (huffman_2val.size() < best_data.size()) {
      best_method = CompressionMethod::RLE_HUFFMAN_2VAL;
      best_data = std::move(huffman_2val);
    }
  }

  // Try Method 9: RLE_HUFFMAN_3VAL (optimized for 3-value blocks with prediction)
  if (num_distinct == 3) {
    auto huffman_3val = compress_rle_huffman_3val(values, count);
    if (huffman_3val.size() < best_data.size()) {
      best_method = CompressionMethod::RLE_HUFFMAN_3VAL;
      best_data = std::move(huffman_3val);
    }
  }

  return {best_method, std::move(best_data)};
}

std::size_t expected_compressed_size(std::size_t num_values, CompressionMethod method) {
  switch (method) {
    case CompressionMethod::RAW_2BIT:
      return (num_values + 3) / 4;

    default:
      return 0;  // Cannot determine without compressing
  }
}

// ============================================================================
// ============================================================================
// CompressedTablebase Creation and Lookup (Stage 3)
// ============================================================================

// Helper: extend runs through tense positions in a block.
// Tense positions (where captures are available) can store any value since
// lookup_wdl_with_search will compute the correct result by searching.
// By extending runs through these positions, we get longer runs for better compression.
static std::vector<Value> extend_tense_positions(
    const Value* values,
    std::size_t count,
    std::size_t block_start_index,
    const Material& m) {

  std::vector<Value> result(count);
  Value prev_value = values[0];  // Start with first value (may be wrong for tense first pos)

  for (std::size_t i = 0; i < count; ++i) {
    std::size_t global_idx = block_start_index + i;
    Board board = index_to_board(global_idx, m);

    if (has_captures(board)) {
      // Tense position - extend previous run
      result[i] = prev_value;
    } else {
      // Normal position - use actual value and update prev
      result[i] = values[i];
      prev_value = values[i];
    }
  }

  return result;
}

CompressedTablebase compress_tablebase(
    const std::vector<Value>& values,
    const Material& m) {

  CompressedTablebase tb;
  tb.material = m;
  tb.num_positions = values.size();
  tb.num_blocks = static_cast<std::uint32_t>((tb.num_positions + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Structure to hold per-block compression results
  struct BlockResult {
    CompressionMethod method;
    std::vector<std::uint8_t> data;
  };
  std::vector<BlockResult> block_results(tb.num_blocks);

  // Compress blocks in parallel
  #pragma omp parallel for schedule(dynamic)
  for (std::uint32_t block_idx = 0; block_idx < tb.num_blocks; ++block_idx) {
    std::size_t block_start = static_cast<std::size_t>(block_idx) * BLOCK_SIZE;
    std::size_t block_end = std::min(block_start + BLOCK_SIZE, values.size());
    std::size_t num_values = block_end - block_start;

    // Extend runs through tense positions for better compression.
    // Tense positions (where captures available) don't need correct values stored
    // since lookup_wdl_with_search will compute them by searching.
    auto extended = extend_tense_positions(
        values.data() + block_start, num_values, block_start, m);

    // Find best compression using the extended values
    auto [method, compressed] = compress_block_best(extended.data(), num_values);

    block_results[block_idx].method = method;
    block_results[block_idx].data = std::move(compressed);
  }

  // Sequentially assemble the compressed tablebase
  tb.block_offsets.reserve(tb.num_blocks);

  // Estimate total size to avoid reallocations
  std::size_t total_data_size = 0;
  for (const auto& br : block_results) {
    total_data_size += 3 + br.data.size();  // 3 bytes header + data
  }
  tb.block_data.reserve(total_data_size);

  for (std::uint32_t block_idx = 0; block_idx < tb.num_blocks; ++block_idx) {
    const auto& br = block_results[block_idx];

    // Record offset
    tb.block_offsets.push_back(static_cast<std::uint32_t>(tb.block_data.size()));

    // Write block header: method (1 byte) + size (2 bytes) + data
    tb.block_data.push_back(static_cast<std::uint8_t>(br.method));
    std::uint16_t size = static_cast<std::uint16_t>(br.data.size());
    tb.block_data.push_back(size & 0xFF);
    tb.block_data.push_back((size >> 8) & 0xFF);

    // Write compressed data
    tb.block_data.insert(tb.block_data.end(), br.data.begin(), br.data.end());
  }

  return tb;
}

Value lookup_compressed(
    const CompressedTablebase& tb,
    std::size_t index) {

  std::uint32_t block_idx = static_cast<std::uint32_t>(index / BLOCK_SIZE);
  std::size_t idx_in_block = index % BLOCK_SIZE;

  // Get block info
  std::uint32_t block_offset = tb.block_offsets[block_idx];
  const std::uint8_t* block_ptr = tb.block_data.data() + block_offset;
  CompressionMethod method = static_cast<CompressionMethod>(block_ptr[0]);
  std::uint16_t compressed_size = block_ptr[1] | (block_ptr[2] << 8);
  const std::uint8_t* data = block_ptr + 3;

  switch (method) {
    case CompressionMethod::RAW_2BIT: {
      std::size_t byte_idx = idx_in_block / 4;
      std::size_t bit_pos = (idx_in_block % 4) * 2;
      return int_to_value((data[byte_idx] >> bit_pos) & 0x3);
    }
    case CompressionMethod::RLE_BINARY_SEARCH:
      return lookup_rle_binary_search(data, compressed_size, idx_in_block);
    case CompressionMethod::DEFAULT_EXCEPTIONS:
      return lookup_default_exceptions(data, compressed_size, idx_in_block);
    case CompressionMethod::RLE_HUFFMAN_2VAL:
      return lookup_rle_huffman_2val(data, compressed_size, idx_in_block);
    case CompressionMethod::RLE_HUFFMAN_3VAL:
      return lookup_rle_huffman_3val(data, compressed_size, idx_in_block);
  }

  std::cerr << "lookup_compressed: unknown compression method "
            << static_cast<int>(method) << "\n";
  std::exit(1);
}

// Look up WDL value from a compressed tablebase, searching through captures.
// Since captures are irreversible, recursion depth is bounded by piece count.
Value lookup_compressed_with_search(
    const Board& b,
    const CompressedTablebase& tb) {

  auto lookup = [&](const Board& pos) -> int {
    Material pos_m = get_material(pos);
    if (!(pos_m == tb.material)) {
      // Material changed - would need sub-tablebase lookup
      return SCORE_DRAW;  // Conservative
    }
    std::size_t idx = board_to_index(pos, tb.material);
    return value_to_score(lookup_compressed(tb, idx));
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}

// ============================================================================
// Compression Statistics (Stage 3)
// ============================================================================

BlockCompressionStats analyze_block_compression(const CompressedTablebase& tb) {
  BlockCompressionStats stats;
  stats.total_blocks = tb.num_blocks;
  stats.uncompressed_size = tb.num_positions;  // 1 byte per value uncompressed

  for (std::uint32_t block_idx = 0; block_idx < tb.num_blocks; ++block_idx) {
    std::uint32_t block_offset = tb.block_offsets[block_idx];
    const std::uint8_t* block_ptr = tb.block_data.data() + block_offset;

    CompressionMethod method = static_cast<CompressionMethod>(block_ptr[0]);
    std::uint16_t compressed_size = block_ptr[1] | (block_ptr[2] << 8);

    if (static_cast<int>(method) < 16) {
      stats.method_counts[static_cast<int>(method)]++;
    }

    // Block header (3 bytes) + compressed data
    stats.compressed_size += 3 + compressed_size;
  }

  return stats;
}

// ============================================================================
// BitWriter/BitReader Implementations (Stage 4)
// ============================================================================

BitWriter::BitWriter() : buffer_(), current_byte_(0), bits_in_byte_(0), bit_count_(0) {
  buffer_.reserve(1024);  // Pre-allocate for typical block sizes
}

void BitWriter::write(std::uint32_t value, int num_bits) {
  bit_count_ += num_bits;

  while (num_bits > 0) {
    int bits_available = 8 - bits_in_byte_;
    int bits_to_write = std::min(num_bits, bits_available);

    // Extract the top bits_to_write bits from value
    std::uint32_t mask = (1u << bits_to_write) - 1;
    std::uint32_t bits = (value >> (num_bits - bits_to_write)) & mask;

    // Add them to the current byte
    current_byte_ |= (bits << (bits_available - bits_to_write));
    bits_in_byte_ += bits_to_write;

    // If byte is full, flush it
    if (bits_in_byte_ == 8) {
      buffer_.push_back(static_cast<std::uint8_t>(current_byte_));
      current_byte_ = 0;
      bits_in_byte_ = 0;
    }

    num_bits -= bits_to_write;
  }
}

std::vector<std::uint8_t> BitWriter::finish() {
  // Flush any remaining bits
  if (bits_in_byte_ > 0) {
    buffer_.push_back(static_cast<std::uint8_t>(current_byte_));
  }
  return std::move(buffer_);
}

BitReader::BitReader(const std::uint8_t* data, std::size_t size)
    : data_(data), size_(size), bit_pos_(0) {}

std::uint32_t BitReader::read(int num_bits) {
  std::uint32_t result = 0;

  while (num_bits > 0) {
    std::size_t byte_idx = bit_pos_ / 8;
    int bit_in_byte = bit_pos_ % 8;

    if (byte_idx >= size_) {
      bit_pos_ += num_bits;
      return result << num_bits;  // Pad with zeros
    }

    int bits_available = 8 - bit_in_byte;
    int bits_to_read = std::min(num_bits, bits_available);

    std::uint32_t mask = (1u << bits_to_read) - 1;
    std::uint32_t bits = (data_[byte_idx] >> (bits_available - bits_to_read)) & mask;

    result = (result << bits_to_read) | bits;
    bit_pos_ += bits_to_read;
    num_bits -= bits_to_read;
  }

  return result;
}

std::uint32_t BitReader::peek(int num_bits) const {
  BitReader copy = *this;
  return copy.read(num_bits);
}

bool BitReader::has_bits(int num_bits) const {
  return (bit_pos_ + num_bits) <= (size_ * 8);
}

// ============================================================================
// Run-Length Statistics Collection (Stage 4)
// ============================================================================

void collect_block_run_statistics(const Value* values, std::size_t count, RunStatistics& stats) {
  if (count == 0) return;

  stats.total_blocks++;
  stats.total_positions += count;

  // Count distinct values and value frequencies
  bool seen[4] = {false, false, false, false};
  int num_distinct = 0;

  for (std::size_t i = 0; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    stats.value_counts[v]++;
    if (!seen[v]) {
      seen[v] = true;
      num_distinct++;
    }
  }

  if (num_distinct >= 1 && num_distinct <= 4) {
    stats.distinct_value_histogram[num_distinct]++;
  }

  // Extract runs
  struct Run {
    std::uint8_t value;
    std::size_t length;
  };
  std::vector<Run> runs;

  std::uint8_t current_value = value_to_int(values[0]);
  std::size_t current_length = 1;

  for (std::size_t i = 1; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (v == current_value) {
      current_length++;
    } else {
      runs.push_back({current_value, current_length});
      current_value = v;
      current_length = 1;
    }
  }
  runs.push_back({current_value, current_length});

  stats.total_runs += runs.size();

  // Run length histogram (log-scale buckets)
  for (const Run& run : runs) {
    int bucket = 0;
    std::size_t len = run.length;
    while (len > 1 && bucket < 15) {
      len >>= 1;
      bucket++;
    }
    stats.run_length_histogram[bucket]++;
  }

  // Prediction accuracy: does run k have same value as run k-2?
  for (std::size_t k = 2; k < runs.size(); ++k) {
    stats.prediction_total++;
    if (runs[k].value == runs[k - 2].value) {
      stats.prediction_correct++;
    }
  }
}

RunStatistics collect_all_tablebase_statistics(const std::string& directory) {
  RunStatistics stats;

  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (!entry.is_regular_file()) continue;

    std::string filename = entry.path().filename().string();
    if (filename.size() < 10 || filename.substr(0, 3) != "tb_" ||
        filename.substr(filename.size() - 4) != ".bin") {
      continue;
    }

    // Parse material from filename: tb_BBWWKK.bin
    std::string mat_str = filename.substr(3, 6);
    if (mat_str.size() != 6) continue;

    Material m;
    m.back_white_pawns = mat_str[0] - '0';
    m.back_black_pawns = mat_str[1] - '0';
    m.other_white_pawns = mat_str[2] - '0';
    m.other_black_pawns = mat_str[3] - '0';
    m.white_queens = mat_str[4] - '0';
    m.black_queens = mat_str[5] - '0';

    std::vector<Value> tb = load_tablebase(m);
    if (tb.empty()) continue;

    // Process in blocks
    for (std::size_t block_start = 0; block_start < tb.size(); block_start += BLOCK_SIZE) {
      std::size_t block_end = std::min(block_start + BLOCK_SIZE, tb.size());
      std::size_t block_count = block_end - block_start;
      collect_block_run_statistics(tb.data() + block_start, block_count, stats);
    }
  }

  return stats;
}

void print_run_statistics(const RunStatistics& stats) {
  std::cout << "=== Run-Length Statistics ===\n";
  std::cout << "Total blocks:    " << stats.total_blocks << "\n";
  std::cout << "Total runs:      " << stats.total_runs << "\n";
  std::cout << "Total positions: " << stats.total_positions << "\n";
  std::cout << "Avg run length:  " << std::fixed << std::setprecision(2)
            << stats.avg_run_length() << "\n";
  std::cout << "Prediction accuracy: " << std::fixed << std::setprecision(1)
            << (100.0 * stats.prediction_accuracy()) << "%\n\n";

  std::cout << "Run length histogram (log buckets):\n";
  for (int i = 0; i < 16; ++i) {
    if (stats.run_length_histogram[i] > 0) {
      int low = (i == 0) ? 1 : (1 << i);
      int high = (1 << (i + 1)) - 1;
      std::cout << "  [" << std::setw(5) << low << "-" << std::setw(5) << high << "]: "
                << stats.run_length_histogram[i] << "\n";
    }
  }

  std::cout << "\nDistinct values per block:\n";
  for (int i = 1; i <= 4; ++i) {
    std::cout << "  " << i << " values: " << stats.distinct_value_histogram[i] << "\n";
  }

  std::cout << "\nValue distribution:\n";
  const char* value_names[] = {"UNKNOWN", "WIN", "LOSS", "DRAW"};
  for (int i = 0; i < 4; ++i) {
    std::cout << "  " << std::setw(7) << value_names[i] << ": " << stats.value_counts[i] << "\n";
  }
}

// ============================================================================
// Optimized Huffman RLE for 2-Value Blocks (Method 8)
// ============================================================================
//
// Format (minimal overhead for 2-value blocks):
//   Byte 0:     val_0 (bits 0-1) | val_1 (bits 2-3) | reserved (bits 4-7)
//   Bytes 1-2:  run_count (uint16_t LE)
//   Bytes 3-4:  bit_stream_bytes (uint16_t LE)
//   Bytes 5+:   Huffman-encoded run lengths (values alternate automatically)
//
// Encoding scheme (optimized from 6-men EGTB statistics):
//   0                       = length 1    (1 bit)   ~46%
//   10                      = length 2    (2 bits)  ~18%
//   110                     = length 3    (3 bits)  ~9%
//   1110                    = length 4    (4 bits)  ~4%
//   11110 + 2 bits          = length 5-8  (7 bits)  ~8%
//   111110 + 3 bits         = length 9-16 (9 bits)  ~5%
//   1111110 + 4 bits        = length 17-32 (11 bits) ~5%
//   11111110 + 5 bits       = length 33-64 (13 bits) ~2%
//   111111110 + 6 bits      = length 65-128 (15 bits) ~1%
//   1111111110 + 7 bits     = length 129-256 (17 bits)
//   11111111110 + 8 bits    = length 257-512 (19 bits)
//   111111111110 + 9 bits   = length 513-1024 (21 bits)
//   1111111111110 + 10 bits = length 1025-2048 (23 bits)
//   11111111111110 + 14 bits = length 2049-16384 (28 bits)
//
// Expected: ~3.5 bits/run, ~0.24 bits/position (vs 2 bits/position for RAW_2BIT)

std::vector<std::uint8_t> compress_rle_huffman_2val(const Value* values, std::size_t count) {
  if (count == 0) return {};

  // Extract runs
  struct Run { std::size_t length; };
  std::vector<Run> runs;
  runs.reserve(count / 10);  // Estimate based on avg run length ~14

  std::uint8_t first_value = value_to_int(values[0]);
  std::uint8_t second_value = first_value;  // Will be set when we find a different value
  std::size_t current_length = 1;

  for (std::size_t i = 1; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (v == value_to_int(values[i - 1])) {
      current_length++;
    } else {
      runs.push_back({current_length});
      if (second_value == first_value && v != first_value) {
        second_value = v;
      }
      current_length = 1;
    }
  }
  runs.push_back({current_length});

  // Handle single-run case (all same value)
  if (runs.size() == 1) {
    std::vector<std::uint8_t> result(2);
    result[0] = first_value & 0x3;
    result[1] = 0;  // Marker for single-value block
    return result;
  }

  // Encode runs using Huffman
  BitWriter writer;

  for (const Run& run : runs) {
    std::size_t len = run.length;
    if (len == 0) len = 1;

    if (len == 1) {
      writer.write(0b0, 1);
    } else if (len == 2) {
      writer.write(0b10, 2);
    } else if (len == 3) {
      writer.write(0b110, 3);
    } else if (len == 4) {
      writer.write(0b1110, 4);
    } else if (len <= 8) {
      writer.write((0b11110u << 2) | (len - 5), 7);
    } else if (len <= 16) {
      writer.write((0b111110u << 3) | (len - 9), 9);
    } else if (len <= 32) {
      writer.write((0b1111110u << 4) | (len - 17), 11);
    } else if (len <= 64) {
      writer.write((0b11111110u << 5) | (len - 33), 13);
    } else if (len <= 128) {
      writer.write((0b111111110u << 6) | (len - 65), 15);
    } else if (len <= 256) {
      writer.write((0b1111111110u << 7) | (len - 129), 17);
    } else if (len <= 512) {
      writer.write((0b11111111110u << 8) | (len - 257), 19);
    } else if (len <= 1024) {
      writer.write((0b111111111110u << 9) | (len - 513), 21);
    } else if (len <= 2048) {
      writer.write((0b1111111111110u << 10) | (len - 1025), 23);
    } else {
      // 2049-16384
      len = std::min(len, std::size_t(16384));
      writer.write(0b11111111111110u, 14);
      writer.write(static_cast<std::uint32_t>(len - 2049), 14);
    }
  }

  auto bits = writer.finish();

  // Build result
  std::vector<std::uint8_t> result;
  result.reserve(5 + bits.size());

  // Header byte: val_0 (bits 0-1) | val_1 (bits 2-3)
  std::uint8_t header = (first_value & 0x3) | ((second_value & 0x3) << 2);
  result.push_back(header);

  // Run count
  std::uint16_t run_count = static_cast<std::uint16_t>(runs.size());
  result.push_back(run_count & 0xFF);
  result.push_back((run_count >> 8) & 0xFF);

  // Bit stream size
  std::uint16_t bit_bytes = static_cast<std::uint16_t>(bits.size());
  result.push_back(bit_bytes & 0xFF);
  result.push_back((bit_bytes >> 8) & 0xFF);

  // Bit stream
  result.insert(result.end(), bits.begin(), bits.end());

  return result;
}

std::vector<Value> decompress_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values) {
  if (num_values == 0 || data_size < 2) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Check for single-value block marker
  if (data_size == 2 && data[1] == 0) {
    std::uint8_t val = data[0] & 0x3;
    return std::vector<Value>(num_values, int_to_value(val));
  }

  if (data_size < 5) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Parse header
  std::uint8_t header = data[0];
  std::uint8_t val_0 = header & 0x3;
  std::uint8_t val_1 = (header >> 2) & 0x3;

  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  if (data_size < static_cast<std::size_t>(5) + bit_bytes) {
    return std::vector<Value>(num_values, Value::UNKNOWN);
  }

  // Decode runs
  BitReader reader(data + 5, bit_bytes);
  std::vector<Value> result;
  result.reserve(num_values);

  std::uint8_t current_value = val_0;

  for (std::uint16_t r = 0; r < run_count && result.size() < num_values; ++r) {
    // Decode run length using Huffman
    std::size_t run_length;

    // Read prefix (count leading 1s until we hit a 0)
    int prefix = 0;
    while (reader.has_bits(1) && reader.read(1) == 1) {
      prefix++;
      if (prefix >= 14) break;
    }

    switch (prefix) {
      case 0: run_length = 1; break;
      case 1: run_length = 2; break;
      case 2: run_length = 3; break;
      case 3: run_length = 4; break;
      case 4: run_length = 5 + reader.read(2); break;
      case 5: run_length = 9 + reader.read(3); break;
      case 6: run_length = 17 + reader.read(4); break;
      case 7: run_length = 33 + reader.read(5); break;
      case 8: run_length = 65 + reader.read(6); break;
      case 9: run_length = 129 + reader.read(7); break;
      case 10: run_length = 257 + reader.read(8); break;
      case 11: run_length = 513 + reader.read(9); break;
      case 12: run_length = 1025 + reader.read(10); break;
      default: run_length = 2049 + reader.read(14); break;
    }

    // Add values
    std::size_t to_add = std::min(run_length, num_values - result.size());
    for (std::size_t i = 0; i < to_add; ++i) {
      result.push_back(int_to_value(current_value));
    }

    // Alternate value for next run
    current_value = (current_value == val_0) ? val_1 : val_0;
  }

  // Pad if needed
  while (result.size() < num_values) {
    result.push_back(Value::UNKNOWN);
  }

  return result;
}

// Lookup a single value from Huffman 2-val encoded data by iterating through runs.
static Value lookup_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                                      std::size_t target_idx) {
  // Check for single-value block marker (2-byte format: value byte + 0x00 marker)
  if (data_size == 2 && data[1] == 0) {
    return int_to_value(data[0] & 0x3);
  }

  // Parse header
  std::uint8_t val_0 = data[0] & 0x3;
  std::uint8_t val_1 = (data[0] >> 2) & 0x3;
  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  // Decode runs until we find target_idx
  BitReader reader(data + 5, bit_bytes);
  std::size_t pos = 0;
  std::uint8_t current_value = val_0;

  for (std::uint16_t r = 0; r < run_count; ++r) {
    // Read prefix (count leading 1s until we hit a 0)
    int prefix = 0;
    while (reader.read(1) == 1) {
      prefix++;
      if (prefix >= 14) break;
    }

    // Decode run length
    std::size_t run_length;
    switch (prefix) {
      case 0: run_length = 1; break;
      case 1: run_length = 2; break;
      case 2: run_length = 3; break;
      case 3: run_length = 4; break;
      case 4: run_length = 5 + reader.read(2); break;
      case 5: run_length = 9 + reader.read(3); break;
      case 6: run_length = 17 + reader.read(4); break;
      case 7: run_length = 33 + reader.read(5); break;
      case 8: run_length = 65 + reader.read(6); break;
      case 9: run_length = 129 + reader.read(7); break;
      case 10: run_length = 257 + reader.read(8); break;
      case 11: run_length = 513 + reader.read(9); break;
      case 12: run_length = 1025 + reader.read(10); break;
      default: run_length = 2049 + reader.read(14); break;
    }

    // Check if target is in this run
    if (target_idx < pos + run_length) {
      return int_to_value(current_value);
    }

    pos += run_length;
    current_value = (current_value == val_0) ? val_1 : val_0;
  }

  std::cerr << "lookup_rle_huffman_2val: target_idx " << target_idx
            << " not found in " << run_count << " runs\n";
  std::exit(1);
}

// ============================================================================
// Optimized Huffman RLE for 3-Value Blocks with Prediction (Method 9)
// ============================================================================
//
// Format (optimized for 3-value blocks with prediction scheme):
//   Byte 0:     val_0 (bits 0-1) | val_1 (bits 2-3) | val_2 (bits 4-5)
//   Bytes 1-2:  run_count (uint16_t LE)
//   Bytes 3-4:  bit_stream_bytes (uint16_t LE)
//   Bytes 5+:   Huffman-encoded symbols
//
// Prediction scheme:
//   - Run 0 and 1: values from header (treated as TRUE predictions)
//   - Run k (k >= 2): predicted value = run[k-2].value
//     - TRUE: prediction correct (96% of cases)
//     - FALSE: prediction wrong (4% of cases)
//
// Scheme B: Combined (prediction, length_bucket) Huffman encoding
// Based on 6-men EGTB statistics, optimized for ~3.24 bits/symbol
//
// TRUE symbols (sorted by frequency):
//   T1:        0           (1 bit)   36.61%
//   T2:        100         (3 bits)  16.38%
//   T3:        1010        (4 bits)   9.94%
//   T5-8:      1011 + 2b   (6 bits)   9.21%
//   T9-16:     11000 + 3b  (8 bits)   6.53%
//   T17-32:    11001 + 4b  (9 bits)   5.79%
//   T4:        11010       (5 bits)   4.65%
//   T33-64:    110110 + 5b (11 bits)  3.24%
//   T65-128:   1110000 + 6b (13 bits) 1.93%
//   T129-256:  11100010 + 7b (15 bits) 0.94%
//   T257-512:  111001000 + 8b (17 bits) 0.52%
//   T513-1024: 1110010100 + 9b (19 bits) 0.20%
//   T1025-2048: 111001100001 + 10b (22 bits)
//   T2049+:    111001100010 + 14b (26 bits)
//
// FALSE symbols:
//   F1:        110111      (6 bits)   2.50%
//   F2:        11100011    (8 bits)   0.57%
//   F3:        111001001   (9 bits)   0.26%
//   F4:        1110010110  (10 bits)  0.12%
//   F5-8:      1110010101 + 2b (12 bits) 0.20%
//   F9-16:     11100101110 + 3b (14 bits) 0.13%
//   F17-32:    11100101111 + 4b (15 bits) 0.13%
//   F33-64:    111001100000 + 5b (17 bits) 0.05%
//   F65+:      1110011000110 + 14b (27 bits)

// Encode a (TRUE, length) symbol using Scheme B
static void encode_true_symbol(BitWriter& writer, std::size_t len) {
  if (len == 1) {
    writer.write(0b0, 1);
  } else if (len == 2) {
    writer.write(0b100, 3);
  } else if (len == 3) {
    writer.write(0b1010, 4);
  } else if (len == 4) {
    writer.write(0b11010, 5);
  } else if (len <= 8) {
    writer.write(0b1011, 4);
    writer.write(static_cast<std::uint32_t>(len - 5), 2);
  } else if (len <= 16) {
    writer.write(0b11000, 5);
    writer.write(static_cast<std::uint32_t>(len - 9), 3);
  } else if (len <= 32) {
    writer.write(0b11001, 5);
    writer.write(static_cast<std::uint32_t>(len - 17), 4);
  } else if (len <= 64) {
    writer.write(0b110110, 6);
    writer.write(static_cast<std::uint32_t>(len - 33), 5);
  } else if (len <= 128) {
    writer.write(0b1110000, 7);
    writer.write(static_cast<std::uint32_t>(len - 65), 6);
  } else if (len <= 256) {
    writer.write(0b11100010, 8);
    writer.write(static_cast<std::uint32_t>(len - 129), 7);
  } else if (len <= 512) {
    writer.write(0b111001000, 9);
    writer.write(static_cast<std::uint32_t>(len - 257), 8);
  } else if (len <= 1024) {
    writer.write(0b1110010100, 10);
    writer.write(static_cast<std::uint32_t>(len - 513), 9);
  } else if (len <= 2048) {
    writer.write(0b111001100001, 12);
    writer.write(static_cast<std::uint32_t>(len - 1025), 10);
  } else {
    // 2049+
    len = std::min(len, std::size_t(16384));
    writer.write(0b111001100010, 12);
    writer.write(static_cast<std::uint32_t>(len - 2049), 14);
  }
}

// Encode a (FALSE, length, which_value) symbol using Scheme B
// which_value: 0 = val_k_minus_1, 1 = third value
static void encode_false_symbol(BitWriter& writer, std::size_t len, int which_value) {
  if (len == 1) {
    writer.write(0b110111, 6);
  } else if (len == 2) {
    writer.write(0b11100011, 8);
  } else if (len == 3) {
    writer.write(0b111001001, 9);
  } else if (len == 4) {
    writer.write(0b1110010110, 10);
  } else if (len <= 8) {
    writer.write(0b1110010101, 10);
    writer.write(static_cast<std::uint32_t>(len - 5), 2);
  } else if (len <= 16) {
    writer.write(0b11100101110, 11);
    writer.write(static_cast<std::uint32_t>(len - 9), 3);
  } else if (len <= 32) {
    writer.write(0b11100101111, 11);
    writer.write(static_cast<std::uint32_t>(len - 17), 4);
  } else if (len <= 64) {
    writer.write(0b111001100000, 12);
    writer.write(static_cast<std::uint32_t>(len - 33), 5);
  } else {
    // 65+
    len = std::min(len, std::size_t(16384));
    writer.write(0b1110011000110, 13);
    writer.write(static_cast<std::uint32_t>(len - 65), 14);
  }
  // Add 1-bit selector for which FALSE value
  writer.write(static_cast<std::uint32_t>(which_value), 1);
}

// Decode symbol returns: (is_true, length, which_false)
// which_false is only valid when is_true == false: 0 = val_k_minus_1, 1 = third
static std::tuple<bool, std::size_t, int> decode_symbol(BitReader& reader) {
  // Helper to read which_false bit for FALSE symbols
  auto read_false = [&reader](std::size_t len) -> std::tuple<bool, std::size_t, int> {
    int which = reader.has_bits(1) ? static_cast<int>(reader.read(1)) : 0;
    return {false, len, which};
  };

  // Prefix codes (from Scheme B design):
  // 0               = T1
  // 100             = T2
  // 1010            = T3
  // 1011 + 2b       = T5-8
  // 11000 + 3b      = T9-16
  // 11001 + 4b      = T17-32
  // 11010           = T4
  // 110110 + 5b     = T33-64
  // 110111 + which  = F1
  // 1110000 + 6b    = T65-128
  // 11100010 + 7b   = T129-256
  // 11100011 + which = F2
  // 111001000 + 8b  = T257-512
  // 111001001 + which = F3
  // 1110010100 + 9b = T513-1024
  // 1110010101 + 2b + which = F5-8
  // 1110010110 + which = F4
  // 11100101110 + 3b + which = F9-16
  // 11100101111 + 4b + which = F17-32
  // 111001100000 + 5b + which = F33-64
  // 111001100001 + 10b = T1025-2048
  // 111001100010 + 14b = T2049+
  // 1110011000110 + 14b + which = F65+

  if (!reader.has_bits(1)) return {true, 1, 0};

  // Bit 0
  if (reader.read(1) == 0) return {true, 1, 0};  // T1

  // Bit 1-2: 1xx
  if (!reader.has_bits(2)) return {true, 1, 0};
  std::uint32_t b12 = reader.read(2);

  if (b12 == 0b00) return {true, 2, 0};  // 100 = T2

  if (b12 == 0b01) {
    // 101x
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 3, 0};  // 1010 = T3
    // 1011 + 2b = T5-8
    return {true, 5 + reader.read(2), 0};
  }

  if (b12 == 0b10) {
    // 110xx
    if (!reader.has_bits(2)) return {true, 1, 0};
    std::uint32_t b34 = reader.read(2);

    if (b34 == 0b00) return {true, 9 + reader.read(3), 0};   // 11000 + 3b = T9-16
    if (b34 == 0b01) return {true, 17 + reader.read(4), 0};  // 11001 + 4b = T17-32
    if (b34 == 0b10) return {true, 4, 0};                     // 11010 = T4

    // 11011x
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 33 + reader.read(5), 0};  // 110110 + 5b = T33-64
    return read_false(1);  // 110111 = F1
  }

  // b12 == 0b11: 111xxxx
  if (!reader.has_bits(4)) return {true, 1, 0};
  std::uint32_t b3456 = reader.read(4);

  if (b3456 == 0b0000) return {true, 65 + reader.read(6), 0};  // 1110000 + 6b = T65-128

  if (b3456 == 0b0001) {
    // 1110001x
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 129 + reader.read(7), 0};  // 11100010 + 7b = T129-256
    return read_false(2);  // 11100011 = F2
  }

  if (b3456 == 0b0010) {
    // 1110010x...
    // Tree structure:
    // bit7=0: 1110010|0x → bit8=0: T257-512, bit8=1: F3
    // bit7=1: 1110010|1xx → bits8-9: 00=T513-1024, 01=F5-8, 10=F4, 11=F9-16/F17-32
    if (!reader.has_bits(1)) return {true, 1, 0};
    std::uint32_t bit7 = reader.read(1);

    if (bit7 == 0) {
      // 11100100x
      if (!reader.has_bits(1)) return {true, 1, 0};
      if (reader.read(1) == 0) return {true, 257 + reader.read(8), 0};  // 111001000 + 8b = T257-512
      return read_false(3);                                              // 111001001 = F3
    }

    // bit7 == 1: 11100101xx
    if (!reader.has_bits(2)) return {true, 1, 0};
    std::uint32_t b89 = reader.read(2);

    if (b89 == 0b00) return {true, 513 + reader.read(9), 0};  // 1110010100 + 9b = T513-1024
    if (b89 == 0b01) return read_false(5 + reader.read(2));   // 1110010101 + 2b = F5-8
    if (b89 == 0b10) return read_false(4);                     // 1110010110 = F4

    // b89 == 0b11: 1110010111x
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return read_false(9 + reader.read(3));   // 11100101110 + 3b = F9-16
    return read_false(17 + reader.read(4));                            // 11100101111 + 4b = F17-32
  }

  if (b3456 == 0b0011) {
    // 1110011xxxxx
    if (!reader.has_bits(5)) return {true, 1, 0};
    std::uint32_t b789ab = reader.read(5);

    if (b789ab == 0b00000) return read_false(33 + reader.read(5));      // 111001100000 + 5b = F33-64
    if (b789ab == 0b00001) return {true, 1025 + reader.read(10), 0};    // 111001100001 + 10b = T1025-2048
    if (b789ab == 0b00010) return {true, 2049 + reader.read(14), 0};    // 111001100010 + 14b = T2049+

    // 1110011000110 + 14b = F65+  (b789ab == 0b00011, then read 1 more bit which must be 0)
    if (b789ab == 0b00011) {
      if (!reader.has_bits(1)) return {true, 1, 0};
      reader.read(1);  // Read and discard the trailing 0
      return read_false(65 + reader.read(14));
    }
  }

  // Fallback (shouldn't reach here in valid data)
  return {true, 1, 0};
}

std::vector<std::uint8_t> compress_rle_huffman_3val(const Value* values, std::size_t count) {
  if (count == 0) return {};

  // Extract runs with values
  struct Run {
    std::uint8_t value;
    std::size_t length;
  };
  std::vector<Run> runs;
  runs.reserve(count / 10);

  // Track distinct values and build runs
  std::uint8_t val_0 = value_to_int(values[0]);
  std::uint8_t val_1 = val_0;  // Will be set when we find second value
  int num_distinct = 1;

  std::size_t current_length = 1;

  for (std::size_t i = 1; i < count; ++i) {
    std::uint8_t v = value_to_int(values[i]);
    if (v == value_to_int(values[i - 1])) {
      current_length++;
    } else {
      runs.push_back({value_to_int(values[i - 1]), current_length});

      // Track distinct values (just need to know count, not all values)
      if (num_distinct == 1 && v != val_0) {
        val_1 = v;
        num_distinct = 2;
      } else if (num_distinct == 2 && v != val_0 && v != val_1) {
        num_distinct = 3;
      }

      current_length = 1;
    }
  }
  runs.push_back({value_to_int(values[count - 1]), current_length});

  // Handle single-run case (all same value)
  if (runs.size() == 1) {
    std::vector<std::uint8_t> result(2);
    result[0] = val_0 & 0x3;
    result[1] = 0;  // Marker for single-value block
    return result;
  }

  // Handle 2-value case (no prediction bits needed, same as 2-value method)
  if (num_distinct <= 2) {
    return compress_rle_huffman_2val(values, count);
  }

  // Header stores actual run values:
  // - run0_val: actual value of run 0
  // - run1_val: actual value of run 1 (guaranteed different from run0_val since runs alternate)
  // - third_val: the third distinct value (the one that's not run0_val or run1_val)
  std::uint8_t run0_val = runs[0].value;
  std::uint8_t run1_val = runs[1].value;

  // Find the third value (the one that's neither run0_val nor run1_val)
  std::uint8_t third_val = 0;
  for (std::size_t i = 0; i < runs.size(); ++i) {
    if (runs[i].value != run0_val && runs[i].value != run1_val) {
      third_val = runs[i].value;
      break;
    }
  }

  // Encode runs using Huffman with prediction
  BitWriter writer;

  // Track state for prediction: val_k_minus_2, val_k_minus_1
  // Bootstrap: pretend k=-2 had run0_val, k=-1 had run1_val
  std::uint8_t val_k_minus_2 = run0_val;
  std::uint8_t val_k_minus_1 = run1_val;

  for (std::size_t k = 0; k < runs.size(); ++k) {
    std::uint8_t run_value = runs[k].value;
    std::size_t run_length = runs[k].length;

    if (k == 0 || k == 1) {
      // Runs 0 and 1: values from header, treated as TRUE predictions
      encode_true_symbol(writer, run_length);
    } else {
      // Run k >= 2: check if prediction (val_k_minus_2) is correct
      if (run_value == val_k_minus_2) {
        encode_true_symbol(writer, run_length);
      } else {
        // FALSE: determine which_value (0 = val_k_minus_1, 1 = third)
        int which_value = (run_value == val_k_minus_1) ? 0 : 1;
        encode_false_symbol(writer, run_length, which_value);
      }
    }

    // Update state for next iteration (for all k, including 0 and 1)
    val_k_minus_2 = val_k_minus_1;
    val_k_minus_1 = run_value;
  }

  auto bits = writer.finish();

  // Build result
  std::vector<std::uint8_t> result;
  result.reserve(5 + bits.size());

  // Header byte: run0_val (bits 0-1) | run1_val (bits 2-3) | third_val (bits 4-5)
  std::uint8_t header = (run0_val & 0x3) | ((run1_val & 0x3) << 2) | ((third_val & 0x3) << 4);
  result.push_back(header);

  // Run count
  std::uint16_t run_count = static_cast<std::uint16_t>(runs.size());
  result.push_back(run_count & 0xFF);
  result.push_back((run_count >> 8) & 0xFF);

  // Bit stream size
  std::uint16_t bit_bytes = static_cast<std::uint16_t>(bits.size());
  result.push_back(bit_bytes & 0xFF);
  result.push_back((bit_bytes >> 8) & 0xFF);

  // Bit stream
  result.insert(result.end(), bits.begin(), bits.end());

  return result;
}

std::vector<Value> decompress_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values) {
  if (num_values == 0 || data_size < 2) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Check for single-value block marker
  if (data_size == 2 && data[1] == 0) {
    std::uint8_t val = data[0] & 0x3;
    return std::vector<Value>(num_values, int_to_value(val));
  }

  if (data_size < 5) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Parse header
  std::uint8_t header = data[0];
  std::uint8_t val_0 = header & 0x3;
  std::uint8_t val_1 = (header >> 2) & 0x3;
  std::uint8_t val_2 = (header >> 4) & 0x3;

  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  if (data_size < static_cast<std::size_t>(5) + bit_bytes) {
    return std::vector<Value>(num_values, Value::UNKNOWN);
  }

  // Decode runs
  BitReader reader(data + 5, bit_bytes);
  std::vector<Value> result;
  result.reserve(num_values);

  // Track state for prediction: val_k_minus_2, val_k_minus_1
  std::uint8_t val_k_minus_2 = val_0;
  std::uint8_t val_k_minus_1 = val_1;

  for (std::uint16_t k = 0; k < run_count && result.size() < num_values; ++k) {
    std::uint8_t run_value;
    std::size_t run_length;

    // Decode symbol using Scheme B
    auto [is_true, length, which_false] = decode_symbol(reader);
    run_length = length;

    if (k == 0) {
      // Run 0: value is val_0 from header
      run_value = val_0;
    } else if (k == 1) {
      // Run 1: value is val_1 from header
      run_value = val_1;
    } else {
      // Run k >= 2: determine value based on prediction result
      if (is_true) {
        // TRUE: prediction correct, value = val_k_minus_2
        run_value = val_k_minus_2;
      } else {
        // FALSE: prediction wrong
        if (which_false == 0) {
          // which_false == 0: value = val_k_minus_1
          run_value = val_k_minus_1;
        } else {
          // which_false == 1: value = third (neither k-2 nor k-1)
          if (val_k_minus_2 != val_0 && val_k_minus_1 != val_0) {
            run_value = val_0;
          } else if (val_k_minus_2 != val_1 && val_k_minus_1 != val_1) {
            run_value = val_1;
          } else {
            run_value = val_2;
          }
        }
      }
    }

    // Add values
    std::size_t to_add = std::min(run_length, num_values - result.size());
    for (std::size_t i = 0; i < to_add; ++i) {
      result.push_back(int_to_value(run_value));
    }

    // Update state for next iteration (for all k, including 0 and 1)
    val_k_minus_2 = val_k_minus_1;
    val_k_minus_1 = run_value;
  }

  // Pad if needed
  while (result.size() < num_values) {
    result.push_back(Value::UNKNOWN);
  }

  return result;
}

// Lookup a single value from Huffman 3-val encoded data by iterating through runs.
static Value lookup_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                                      std::size_t target_idx) {
  // Check for single-value block marker (2-byte format: value byte + 0x00 marker)
  if (data_size == 2 && data[1] == 0) {
    return int_to_value(data[0] & 0x3);
  }

  // Parse header
  std::uint8_t val_0 = data[0] & 0x3;
  std::uint8_t val_1 = (data[0] >> 2) & 0x3;
  std::uint8_t val_2 = (data[0] >> 4) & 0x3;
  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  // Decode runs until we find target_idx
  BitReader reader(data + 5, bit_bytes);
  std::size_t pos = 0;

  // Track state for prediction
  std::uint8_t val_k_minus_2 = val_0;
  std::uint8_t val_k_minus_1 = val_1;

  for (std::uint16_t k = 0; k < run_count; ++k) {
    // Decode symbol using Scheme B
    auto [is_true, run_length, which_false] = decode_symbol(reader);

    std::uint8_t run_value;
    if (k == 0) {
      run_value = val_0;
    } else if (k == 1) {
      run_value = val_1;
    } else if (is_true) {
      run_value = val_k_minus_2;
    } else if (which_false == 0) {
      run_value = val_k_minus_1;
    } else {
      // Third value (neither k-2 nor k-1)
      if (val_k_minus_2 != val_0 && val_k_minus_1 != val_0) {
        run_value = val_0;
      } else if (val_k_minus_2 != val_1 && val_k_minus_1 != val_1) {
        run_value = val_1;
      } else {
        run_value = val_2;
      }
    }

    // Check if target is in this run
    if (target_idx < pos + run_length) {
      return int_to_value(run_value);
    }

    pos += run_length;
    val_k_minus_2 = val_k_minus_1;
    val_k_minus_1 = run_value;
  }

  std::cerr << "lookup_rle_huffman_3val: target_idx " << target_idx
            << " not found in " << run_count << " runs\n";
  std::exit(1);
}

// ============================================================================
// Compressed Tablebase File I/O (Stage 5)
// ============================================================================

bool save_compressed_tablebase(const CompressedTablebase& tb, const std::string& filename) {
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    return false;
  }

  // Write magic number
  file.write(CWDL_MAGIC, 4);

  // Write version
  file.write(reinterpret_cast<const char*>(&CWDL_VERSION), 1);

  // Write material (6 bytes)
  std::uint8_t mat[6] = {
    static_cast<std::uint8_t>(tb.material.back_white_pawns),
    static_cast<std::uint8_t>(tb.material.back_black_pawns),
    static_cast<std::uint8_t>(tb.material.other_white_pawns),
    static_cast<std::uint8_t>(tb.material.other_black_pawns),
    static_cast<std::uint8_t>(tb.material.white_queens),
    static_cast<std::uint8_t>(tb.material.black_queens)
  };
  file.write(reinterpret_cast<const char*>(mat), 6);

  // Write num_positions (8 bytes, little-endian) - v2 format
  file.write(reinterpret_cast<const char*>(&tb.num_positions), 8);

  // Write num_blocks (4 bytes, little-endian)
  file.write(reinterpret_cast<const char*>(&tb.num_blocks), 4);

  // Write block offsets (4 bytes each)
  file.write(reinterpret_cast<const char*>(tb.block_offsets.data()),
             tb.block_offsets.size() * sizeof(std::uint32_t));

  // Write block data
  file.write(reinterpret_cast<const char*>(tb.block_data.data()),
             tb.block_data.size());

  return file.good();
}

CompressedTablebase load_compressed_tablebase(const std::string& filename) {
  CompressedTablebase tb;

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return tb;  // Return empty tablebase on error
  }

  // Read and verify magic number
  char magic[4];
  file.read(magic, 4);
  if (std::memcmp(magic, CWDL_MAGIC, 4) != 0) {
    return tb;  // Invalid magic
  }

  // Read and verify version
  std::uint8_t version;
  file.read(reinterpret_cast<char*>(&version), 1);
  if (version != CWDL_VERSION) {
    return tb;  // Unsupported version
  }

  // Read material (6 bytes)
  std::uint8_t mat[6];
  file.read(reinterpret_cast<char*>(mat), 6);
  tb.material.back_white_pawns = mat[0];
  tb.material.back_black_pawns = mat[1];
  tb.material.other_white_pawns = mat[2];
  tb.material.other_black_pawns = mat[3];
  tb.material.white_queens = mat[4];
  tb.material.black_queens = mat[5];

  // Read num_positions (8 bytes in v2 format)
  file.read(reinterpret_cast<char*>(&tb.num_positions), 8);

  // Read num_blocks
  file.read(reinterpret_cast<char*>(&tb.num_blocks), 4);

  // Read block offsets
  tb.block_offsets.resize(tb.num_blocks);
  file.read(reinterpret_cast<char*>(tb.block_offsets.data()),
            tb.num_blocks * sizeof(std::uint32_t));

  // Calculate block data size and read it
  // Block data starts after header (4+1+6+8+4 = 23 bytes) + offsets (4*num_blocks)
  std::streampos current_pos = file.tellg();
  file.seekg(0, std::ios::end);
  std::streampos end_pos = file.tellg();
  std::size_t block_data_size = static_cast<std::size_t>(end_pos - current_pos);

  file.seekg(current_pos);
  tb.block_data.resize(block_data_size);
  file.read(reinterpret_cast<char*>(tb.block_data.data()), block_data_size);

  if (!file.good() && !file.eof()) {
    // Read error - return empty tablebase
    return CompressedTablebase{};
  }

  return tb;
}

// ============================================================================
// CompressedTablebaseManager Implementation (Public API)
// ============================================================================

CompressedTablebaseManager::CompressedTablebaseManager(const std::string& directory)
    : directory_(directory) {}

void CompressedTablebaseManager::clear() {
  tb_cache_.clear();
}

CompressedTablebase* CompressedTablebaseManager::load_or_get(const Material& m, bool warn_if_missing) {
  auto it = tb_cache_.find(m);
  if (it != tb_cache_.end()) {
    return it->second.empty() ? nullptr : &it->second;
  }

  // Build filename: cwdl_BBWWKK.bin
  char filename[256];
  std::snprintf(filename, sizeof(filename), "%s/cwdl_%d%d%d%d%d%d.bin",
                directory_.c_str(),
                m.back_white_pawns, m.back_black_pawns,
                m.other_white_pawns, m.other_black_pawns,
                m.white_queens, m.black_queens);

  CompressedTablebase tb = load_compressed_tablebase(filename);

  // Warn once if tablebase is missing (for debugging dependency issues)
  if (tb.empty() && warn_if_missing) {
    static std::unordered_set<Material> warned;
    static std::mutex warn_mutex;
    std::lock_guard<std::mutex> lock(warn_mutex);
    if (warned.find(m) == warned.end()) {
      warned.insert(m);
      std::cerr << "\n  WARNING: Missing dependency tablebase: " << filename << std::endl;
    }
  }

  tb_cache_[m] = std::move(tb);

  return tb_cache_[m].empty() ? nullptr : &tb_cache_[m];
}

const CompressedTablebase* CompressedTablebaseManager::get_tablebase(const Material& m) {
  // Don't warn when just checking for existence
  return load_or_get(m, false);
}

// Look up WDL value, searching through capture positions as needed.
// Since captures are irreversible, the recursion depth is bounded by
// the number of pieces on the board (at most ~24).
Value CompressedTablebaseManager::lookup_wdl(const Board& b) {
  Material m = get_material(b);

  // Terminal: opponent has no pieces
  if (m.white_pieces() == 0) {
    return Value::LOSS;
  }

  CompressedTablebase* tb = load_or_get(m, true);
  if (!tb) {
    return Value::UNKNOWN;
  }

  auto lookup = [this](const Board& pos) -> int {
    Material pos_m = get_material(pos);
    CompressedTablebase* pos_tb = load_or_get(pos_m, false);
    if (!pos_tb) {
      return SCORE_DRAW;  // Missing tablebase - conservative
    }
    std::size_t idx = board_to_index(pos, pos_m);
    return value_to_score(lookup_compressed(*pos_tb, idx));
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}
