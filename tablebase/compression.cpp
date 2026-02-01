#include "compression.hpp"
#include "../core/movegen.hpp"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_set>

// ============================================================================
// Helper Functions for WDL Lookup with Search
// ============================================================================

// Score constants for negamax search through captures
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

// Negamax search with alpha-beta pruning through capture positions.
// The Lookup callable returns an int score for quiet positions.
// Max depth is bounded by piece count (captures remove pieces).
constexpr int MAX_NEGAMAX_DEPTH = 32;

template<typename Lookup>
static int negamax(const Board& b, int alpha, int beta, Lookup&& lookup, int depth = 0) {
  // Quiet position: use the lookup function
  if (!has_captures(b)) {
    return lookup(b);
  }

  // Safety limit to prevent stack overflow
  if (depth >= MAX_NEGAMAX_DEPTH) {
    std::cerr << "\nFATAL: negamax depth " << depth << " exceeded limit!\n";
    std::exit(1);
  }

  // Capture position: search through forced moves
  MoveList moves;
  generateMoves(b, moves);

  if (moves.empty()) {
    return SCORE_LOSS;
  }

  for (const Move& move : moves) {
    Board next = makeMove(b, move);

    // Terminal: captured all opponent pieces
    if (next.white == 0) {
      return SCORE_WIN;
    }

    int score = -negamax(next, -beta, -alpha, lookup, depth + 1);

    if (score >= beta) {
      return score;  // Beta cutoff
    }
    alpha = std::max(alpha, score);
  }

  return alpha;
}

// ============================================================================
// Value Encoding/Decoding
// ============================================================================

namespace {

// Map Value enum to small integers for compact storage.
// We use: WIN=0, DRAW=1, LOSS=2, UNKNOWN=3
inline Value int_to_value(std::uint8_t i) {
  switch (i) {
    case 0: return Value::WIN;
    case 1: return Value::DRAW;
    case 2: return Value::LOSS;
    case 3: return Value::UNKNOWN;
    default:
      std::cerr << "int_to_value: invalid value " << static_cast<int>(i) << "\n";
      std::exit(1);
  }
}

// ============================================================================
// Method 0: RAW_2BIT Decompression
// ============================================================================

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
// Method 3: RLE_BINARY_SEARCH Decompression and Lookup
// ============================================================================

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

Value lookup_rle_binary_search(const std::uint8_t* data, std::size_t data_size, std::size_t index) {
  std::size_t num_records = data_size / 2;

  // Binary search to find the record that covers this index.
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

  std::uint16_t record = data[(lo - 1) * 2] | (data[(lo - 1) * 2 + 1] << 8);
  std::uint8_t val = (record >> 14) & 0x3;
  return int_to_value(val);
}

// ============================================================================
// Method 2: DEFAULT_EXCEPTIONS Decompression and Lookup
// ============================================================================

std::vector<Value> decompress_default_exceptions(const std::uint8_t* data, std::size_t data_size, std::size_t num_values) {
  if (data_size < 3) return std::vector<Value>(num_values, Value::UNKNOWN);

  std::uint8_t default_value = data[0] & 0x3;
  std::uint16_t num_exceptions = data[1] | (data[2] << 8);

  std::vector<Value> result(num_values, int_to_value(default_value));

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

Value lookup_default_exceptions(const std::uint8_t* data, std::size_t /*data_size*/, std::size_t index) {
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
      std::uint8_t val = (entry >> 14) & 0x3;
      return int_to_value(val);
    } else if (exc_idx < index) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  return int_to_value(default_value);
}

} // anonymous namespace

// ============================================================================
// BitReader Implementation
// ============================================================================

BitReader::BitReader(const std::uint8_t* data, std::size_t size)
    : data_(data), size_(size), bit_pos_(0) {}

std::uint32_t BitReader::read(int num_bits) {
  std::uint32_t result = 0;

  while (num_bits > 0) {
    std::size_t byte_idx = bit_pos_ / 8;
    int bit_in_byte = bit_pos_ % 8;

    if (byte_idx >= size_) {
      bit_pos_ += num_bits;
      return result << num_bits;
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
// Method 8: RLE_HUFFMAN_2VAL Decompression and Lookup
// ============================================================================

namespace {

std::vector<Value> decompress_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values) {
  if (num_values == 0 || data_size < 2) return std::vector<Value>(num_values, Value::UNKNOWN);

  // Check for single-value block marker
  if (data_size == 2 && data[1] == 0) {
    std::uint8_t val = data[0] & 0x3;
    return std::vector<Value>(num_values, int_to_value(val));
  }

  if (data_size < 5) return std::vector<Value>(num_values, Value::UNKNOWN);

  std::uint8_t header = data[0];
  std::uint8_t val_0 = header & 0x3;
  std::uint8_t val_1 = (header >> 2) & 0x3;

  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  if (data_size < static_cast<std::size_t>(5) + bit_bytes) {
    return std::vector<Value>(num_values, Value::UNKNOWN);
  }

  BitReader reader(data + 5, bit_bytes);
  std::vector<Value> result;
  result.reserve(num_values);

  std::uint8_t current_value = val_0;

  for (std::uint16_t r = 0; r < run_count && result.size() < num_values; ++r) {
    int prefix = 0;
    while (reader.has_bits(1) && reader.read(1) == 1) {
      prefix++;
      if (prefix >= 14) break;
    }

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

    std::size_t to_add = std::min(run_length, num_values - result.size());
    for (std::size_t i = 0; i < to_add; ++i) {
      result.push_back(int_to_value(current_value));
    }

    current_value = (current_value == val_0) ? val_1 : val_0;
  }

  while (result.size() < num_values) {
    result.push_back(Value::UNKNOWN);
  }

  return result;
}

Value lookup_rle_huffman_2val(const std::uint8_t* data, std::size_t data_size,
                               std::size_t target_idx) {
  if (data_size == 2 && data[1] == 0) {
    return int_to_value(data[0] & 0x3);
  }

  std::uint8_t val_0 = data[0] & 0x3;
  std::uint8_t val_1 = (data[0] >> 2) & 0x3;
  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  BitReader reader(data + 5, bit_bytes);
  std::size_t pos = 0;
  std::uint8_t current_value = val_0;

  for (std::uint16_t r = 0; r < run_count; ++r) {
    int prefix = 0;
    while (reader.read(1) == 1) {
      prefix++;
      if (prefix >= 14) break;
    }

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
// Method 9: RLE_HUFFMAN_3VAL Decompression and Lookup
// ============================================================================

// Decode symbol returns: (is_true, length, which_false)
static std::tuple<bool, std::size_t, int> decode_symbol(BitReader& reader) {
  auto read_false = [&reader](std::size_t len) -> std::tuple<bool, std::size_t, int> {
    int which = reader.has_bits(1) ? static_cast<int>(reader.read(1)) : 0;
    return {false, len, which};
  };

  if (!reader.has_bits(1)) return {true, 1, 0};

  if (reader.read(1) == 0) return {true, 1, 0};  // T1

  if (!reader.has_bits(2)) return {true, 1, 0};
  std::uint32_t b12 = reader.read(2);

  if (b12 == 0b00) return {true, 2, 0};  // T2

  if (b12 == 0b01) {
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 3, 0};  // T3
    return {true, 5 + reader.read(2), 0};  // T5-8
  }

  if (b12 == 0b10) {
    if (!reader.has_bits(2)) return {true, 1, 0};
    std::uint32_t b34 = reader.read(2);

    if (b34 == 0b00) return {true, 9 + reader.read(3), 0};
    if (b34 == 0b01) return {true, 17 + reader.read(4), 0};
    if (b34 == 0b10) return {true, 4, 0};

    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 33 + reader.read(5), 0};
    return read_false(1);  // F1
  }

  // b12 == 0b11
  if (!reader.has_bits(4)) return {true, 1, 0};
  std::uint32_t b3456 = reader.read(4);

  if (b3456 == 0b0000) return {true, 65 + reader.read(6), 0};

  if (b3456 == 0b0001) {
    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return {true, 129 + reader.read(7), 0};
    return read_false(2);  // F2
  }

  if (b3456 == 0b0010) {
    if (!reader.has_bits(1)) return {true, 1, 0};
    std::uint32_t bit7 = reader.read(1);

    if (bit7 == 0) {
      if (!reader.has_bits(1)) return {true, 1, 0};
      if (reader.read(1) == 0) return {true, 257 + reader.read(8), 0};
      return read_false(3);  // F3
    }

    if (!reader.has_bits(2)) return {true, 1, 0};
    std::uint32_t b89 = reader.read(2);

    if (b89 == 0b00) return {true, 513 + reader.read(9), 0};
    if (b89 == 0b01) return read_false(5 + reader.read(2));  // F5-8
    if (b89 == 0b10) return read_false(4);  // F4

    if (!reader.has_bits(1)) return {true, 1, 0};
    if (reader.read(1) == 0) return read_false(9 + reader.read(3));  // F9-16
    return read_false(17 + reader.read(4));  // F17-32
  }

  if (b3456 == 0b0011) {
    if (!reader.has_bits(5)) return {true, 1, 0};
    std::uint32_t b789ab = reader.read(5);

    if (b789ab == 0b00000) return read_false(33 + reader.read(5));  // F33-64
    if (b789ab == 0b00001) return {true, 1025 + reader.read(10), 0};  // T1025-2048
    if (b789ab == 0b00010) return {true, 2049 + reader.read(14), 0};  // T2049+

    if (b789ab == 0b00011) {
      if (!reader.has_bits(1)) return {true, 1, 0};
      reader.read(1);
      return read_false(65 + reader.read(14));  // F65+
    }
  }

  return {true, 1, 0};
}

std::vector<Value> decompress_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                                                std::size_t num_values) {
  if (num_values == 0 || data_size < 2) return std::vector<Value>(num_values, Value::UNKNOWN);

  if (data_size == 2 && data[1] == 0) {
    std::uint8_t val = data[0] & 0x3;
    return std::vector<Value>(num_values, int_to_value(val));
  }

  if (data_size < 5) return std::vector<Value>(num_values, Value::UNKNOWN);

  std::uint8_t header = data[0];
  std::uint8_t val_0 = header & 0x3;
  std::uint8_t val_1 = (header >> 2) & 0x3;
  std::uint8_t val_2 = (header >> 4) & 0x3;

  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  if (data_size < static_cast<std::size_t>(5) + bit_bytes) {
    return std::vector<Value>(num_values, Value::UNKNOWN);
  }

  BitReader reader(data + 5, bit_bytes);
  std::vector<Value> result;
  result.reserve(num_values);

  std::uint8_t val_k_minus_2 = val_0;
  std::uint8_t val_k_minus_1 = val_1;

  for (std::uint16_t k = 0; k < run_count && result.size() < num_values; ++k) {
    std::uint8_t run_value;
    auto [is_true, run_length, which_false] = decode_symbol(reader);

    if (k == 0) {
      run_value = val_0;
    } else if (k == 1) {
      run_value = val_1;
    } else {
      if (is_true) {
        run_value = val_k_minus_2;
      } else {
        if (which_false == 0) {
          run_value = val_k_minus_1;
        } else {
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

    std::size_t to_add = std::min(run_length, num_values - result.size());
    for (std::size_t i = 0; i < to_add; ++i) {
      result.push_back(int_to_value(run_value));
    }

    val_k_minus_2 = val_k_minus_1;
    val_k_minus_1 = run_value;
  }

  while (result.size() < num_values) {
    result.push_back(Value::UNKNOWN);
  }

  return result;
}

Value lookup_rle_huffman_3val(const std::uint8_t* data, std::size_t data_size,
                               std::size_t target_idx) {
  if (data_size == 2 && data[1] == 0) {
    return int_to_value(data[0] & 0x3);
  }

  std::uint8_t val_0 = data[0] & 0x3;
  std::uint8_t val_1 = (data[0] >> 2) & 0x3;
  std::uint8_t val_2 = (data[0] >> 4) & 0x3;
  std::uint16_t run_count = data[1] | (data[2] << 8);
  std::uint16_t bit_bytes = data[3] | (data[4] << 8);

  BitReader reader(data + 5, bit_bytes);
  std::size_t pos = 0;

  std::uint8_t val_k_minus_2 = val_0;
  std::uint8_t val_k_minus_1 = val_1;

  for (std::uint16_t k = 0; k < run_count; ++k) {
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
      if (val_k_minus_2 != val_0 && val_k_minus_1 != val_0) {
        run_value = val_0;
      } else if (val_k_minus_2 != val_1 && val_k_minus_1 != val_1) {
        run_value = val_1;
      } else {
        run_value = val_2;
      }
    }

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

} // anonymous namespace

// ============================================================================
// Block Decompression
// ============================================================================

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

// ============================================================================
// Compressed Tablebase Lookup
// ============================================================================

Value lookup_compressed(
    const CompressedTablebase& tb,
    std::size_t index) {

  std::uint32_t block_idx = static_cast<std::uint32_t>(index / BLOCK_SIZE);
  std::size_t idx_in_block = index % BLOCK_SIZE;

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

Value lookup_compressed_with_search(
    const Board& b,
    const CompressedTablebase& tb) {

  auto lookup = [&](const Board& pos) -> int {
    Material pos_m = get_material(pos);
    if (!(pos_m == tb.material)) {
      return SCORE_DRAW;  // Material changed - would need sub-tablebase lookup
    }
    std::size_t idx = board_to_index(pos, tb.material);
    return value_to_score(lookup_compressed(tb, idx));
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}

// ============================================================================
// Compressed Tablebase File I/O
// ============================================================================

CompressedTablebase load_compressed_tablebase(const std::string& filename) {
  CompressedTablebase tb;

  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    return tb;  // Return empty tablebase on error
  }

  // Read and verify magic number
  char magic[4];
  file.read(magic, 4);
  if (!file || std::memcmp(magic, CWDL_MAGIC, 4) != 0) {
    return tb;
  }

  // Read and verify version
  std::uint8_t version;
  file.read(reinterpret_cast<char*>(&version), 1);
  if (!file || version != CWDL_VERSION) {
    return tb;
  }

  // Read material (6 bytes)
  std::uint8_t mat[6];
  file.read(reinterpret_cast<char*>(mat), 6);
  if (!file) return tb;

  tb.material.back_white_pawns = mat[0];
  tb.material.back_black_pawns = mat[1];
  tb.material.other_white_pawns = mat[2];
  tb.material.other_black_pawns = mat[3];
  tb.material.white_queens = mat[4];
  tb.material.black_queens = mat[5];

  // Read num_positions (8 bytes in v2 format)
  file.read(reinterpret_cast<char*>(&tb.num_positions), 8);
  if (!file) return tb;

  // Read num_blocks
  file.read(reinterpret_cast<char*>(&tb.num_blocks), 4);
  if (!file) return tb;

  // Read block offsets
  tb.block_offsets.resize(tb.num_blocks);
  file.read(reinterpret_cast<char*>(tb.block_offsets.data()),
            tb.num_blocks * sizeof(std::uint32_t));
  if (!file) return tb;

  // Read block data
  std::streampos current_pos = file.tellg();
  file.seekg(0, std::ios::end);
  std::streampos end_pos = file.tellg();
  std::size_t block_data_size = static_cast<std::size_t>(end_pos - current_pos);

  file.seekg(current_pos);
  tb.block_data.resize(block_data_size);
  file.read(reinterpret_cast<char*>(tb.block_data.data()), block_data_size);

  return tb;
}

// ============================================================================
// CompressedTablebaseManager Implementation
// ============================================================================

CompressedTablebaseManager::CompressedTablebaseManager(const std::string& directory)
    : directory_(directory) {}

void CompressedTablebaseManager::clear() {
  tb_cache_.clear();
}

void CompressedTablebaseManager::preload(int max_pieces) {
  std::cout << "Preloading compressed WDL tables..." << std::flush;
  int loaded = 0;

  for (int total = 2; total <= max_pieces; ++total) {
    for (int wq = 0; wq <= total; ++wq) {
      for (int bq = 0; bq <= total - wq; ++bq) {
        for (int wp = 0; wp <= total - wq - bq; ++wp) {
          int bp = total - wq - bq - wp;
          if (bp < 0) continue;

          for (int bwp = 0; bwp <= std::min(wp, 4); ++bwp) {
            int owp = wp - bwp;
            for (int bbp = 0; bbp <= std::min(bp, 4); ++bbp) {
              int obp = bp - bbp;

              Material m{bwp, bbp, owp, obp, wq, bq};

              if (m.white_pieces() == 0 || m.black_pieces() == 0) continue;
              if (tb_cache_.count(m)) continue;

              char filename[256];
              std::snprintf(filename, sizeof(filename), "%s/cwdl_%d%d%d%d%d%d.bin",
                            directory_.c_str(),
                            m.back_white_pawns, m.back_black_pawns,
                            m.other_white_pawns, m.other_black_pawns,
                            m.white_queens, m.black_queens);

              if (std::filesystem::exists(filename)) {
                CompressedTablebase tb = load_compressed_tablebase(filename);
                tb_cache_[m] = std::move(tb);
                loaded++;
              } else {
                tb_cache_[m] = CompressedTablebase{};
              }
            }
          }
        }
      }
    }
  }
  std::cout << " loaded " << loaded << " tables\n";
  preloaded_ = true;
}

CompressedTablebase* CompressedTablebaseManager::load_or_get(const Material& m, bool warn_if_missing) {
  auto it = tb_cache_.find(m);
  if (it != tb_cache_.end()) {
    return it->second.empty() ? nullptr : &it->second;
  }

  char filename[256];
  std::snprintf(filename, sizeof(filename), "%s/cwdl_%d%d%d%d%d%d.bin",
                directory_.c_str(),
                m.back_white_pawns, m.back_black_pawns,
                m.other_white_pawns, m.other_black_pawns,
                m.white_queens, m.black_queens);

  CompressedTablebase tb = load_compressed_tablebase(filename);

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
  return load_or_get(m, false);
}

const CompressedTablebase* CompressedTablebaseManager::get_preloaded(const Material& m) const {
  auto it = tb_cache_.find(m);
  if (it != tb_cache_.end() && !it->second.empty()) {
    return &it->second;
  }
  return nullptr;
}

Value CompressedTablebaseManager::lookup_wdl(const Board& b) {
  if (preloaded_) {
    return lookup_wdl_preloaded(b);
  }

  Material m = get_material(b);

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
      return SCORE_DRAW;
    }
    std::size_t idx = board_to_index(pos, pos_m);
    return value_to_score(lookup_compressed(*pos_tb, idx));
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}

Value CompressedTablebaseManager::lookup_wdl_preloaded(const Board& b) const {
  Material m = get_material(b);

  if (m.white_pieces() == 0) {
    return Value::LOSS;
  }

  const CompressedTablebase* tb = get_preloaded(m);
  if (!tb) {
    return Value::UNKNOWN;
  }

  auto lookup = [this](const Board& pos) -> int {
    Material pos_m = get_material(pos);
    const CompressedTablebase* pos_tb = get_preloaded(pos_m);
    if (!pos_tb) {
      return SCORE_DRAW;
    }
    std::size_t idx = board_to_index(pos, pos_m);
    return value_to_score(lookup_compressed(*pos_tb, idx));
  };

  int score = negamax(b, SCORE_LOSS, SCORE_WIN, lookup);
  return score_to_value(score);
}
