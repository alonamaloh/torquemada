#pragma once

#include "../core/board.hpp"
#include "tablebase.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>

// ============================================================================
// Block-Based Compression Constants and Types
// ============================================================================

// Block size for compression (16384 positions per block)
constexpr std::size_t BLOCK_SIZE = 16384;

// Compression method identifiers
enum class CompressionMethod : std::uint8_t {
  RAW_2BIT = 0,           // 2 bits per value, 4 per byte (baseline)
  DEFAULT_EXCEPTIONS = 2, // Default value + sorted exception list
  RLE_BINARY_SEARCH = 3,  // Run-length encoding with binary search
  RLE_HUFFMAN_2VAL = 8,   // Optimized Huffman RLE for 2-value blocks
  RLE_HUFFMAN_3VAL = 9,   // Optimized Huffman RLE for 3-value blocks with prediction
};

// Compressed tablebase structure
struct CompressedTablebase {
  Material material{};
  std::uint64_t num_positions = 0;  // 64-bit to support 10+ piece endgames
  std::uint32_t num_blocks = 0;     // Max ~1M blocks even for 16B positions
  std::vector<std::uint32_t> block_offsets;  // Offset to each block's data
  std::vector<std::uint8_t> block_data;      // Concatenated compressed blocks

  // Get number of positions
  std::size_t size() const { return num_positions; }

  // Check if empty
  bool empty() const { return num_positions == 0; }
};

// ============================================================================
// Block Decompression
// ============================================================================

// Decompress a block of data using the specified method.
// Returns the decompressed values.
std::vector<Value> decompress_block(
    const std::uint8_t* data,
    std::size_t data_size,
    std::size_t num_values,
    CompressionMethod method);

// ============================================================================
// CompressedTablebase Lookup
// ============================================================================

// Look up a single value in a compressed tablebase.
// Decompresses the relevant block on demand (stateless, thread-safe).
Value lookup_compressed(
    const CompressedTablebase& tb,
    std::size_t index);

// Look up a value with search for don't-care positions.
// Combines compressed lookup with search through capture sequences.
Value lookup_compressed_with_search(
    const Board& b,
    const CompressedTablebase& tb);

// ============================================================================
// File Format Constants
// ============================================================================

constexpr char CWDL_MAGIC[4] = {'C', 'W', 'D', 'L'};
constexpr std::uint8_t CWDL_VERSION = 2;  // v2: 64-bit num_positions

// ============================================================================
// Compressed Tablebase File I/O
// ============================================================================

// Load a compressed tablebase from a file.
// Returns an empty tablebase (num_positions == 0) on error.
CompressedTablebase load_compressed_tablebase(const std::string& filename);

// ============================================================================
// BitReader for Huffman Decoding
// ============================================================================

// Bit-level reader for Huffman decoding
class BitReader {
public:
  BitReader(const std::uint8_t* data, std::size_t size);

  // Read bits (returns right-aligned value)
  std::uint32_t read(int num_bits);

  // Peek at next bits without consuming
  std::uint32_t peek(int num_bits) const;

  // Check if more bits available
  bool has_bits(int num_bits) const;

  // Current bit position
  std::size_t bit_pos() const { return bit_pos_; }

private:
  const std::uint8_t* data_;
  std::size_t size_;
  std::size_t bit_pos_ = 0;
};

// ============================================================================
// Compressed Tablebase Manager (Public API)
// ============================================================================

// Manager for compressed tablebases that provides a clean API for looking up
// WDL values. Handles:
// - Loading compressed tablebases on demand
// - Searching through tense (capture) positions
// - Looking up sub-tablebases when material changes after captures
//
// Usage example:
//   CompressedTablebaseManager manager("./tablebases");
//   Value result = manager.lookup_wdl(board);
//
class CompressedTablebaseManager {
public:
  // Construct with directory containing compressed tablebases (cwdl_*.bin)
  explicit CompressedTablebaseManager(const std::string& directory);

  // Look up the WDL value for a position.
  // Handles tense positions by searching through forced capture sequences.
  // Returns Value::UNKNOWN if the tablebase is not available.
  Value lookup_wdl(const Board& board);

  // Thread-safe lookup using only preloaded tables (const, no lazy loading).
  // Call preload() first, then use this from multiple threads.
  Value lookup_wdl_preloaded(const Board& board) const;

  // Get a loaded tablebase (or nullptr if not available)
  const CompressedTablebase* get_tablebase(const Material& m);

  // Get a preloaded tablebase (read-only, thread-safe after preload).
  // Returns nullptr if not preloaded. Does NOT attempt to load.
  const CompressedTablebase* get_preloaded(const Material& m) const;

  // Clear loaded tablebases
  void clear();

  // Preload all tablebases up to max_pieces.
  // Call this before parallel use to avoid any locking/caching overhead.
  void preload(int max_pieces = 7);

  // Get the directory being used
  const std::string& directory() const { return directory_; }

private:
  std::string directory_;
  std::unordered_map<Material, CompressedTablebase> tb_cache_;
  bool preloaded_ = false;  // Set by preload(), makes lookup_wdl thread-safe

  // Load a compressed tablebase (or return cached)
  CompressedTablebase* load_or_get(const Material& m, bool warn_if_missing = true);
};
