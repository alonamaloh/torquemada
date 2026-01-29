#pragma once

#include "../core/board.hpp"
#include "tablebase.hpp"
#include <cstdint>
#include <vector>
#include <unordered_map>

// ============================================================================
// Don't-Care Position Detection (Stage 1)
// ============================================================================

// A position is "don't care" if:
// 1. It has captures available (forced move), OR
// 2. The opponent would have captures after any quiet move we make
//
// For such positions, we don't need to store the WDL value - we can compute
// it by searching through the forced sequence until we reach a "real" position.
bool is_dont_care(const Board& b);

// ============================================================================
// Statistics for Compression Analysis
// ============================================================================

struct CompressionStats {
  std::size_t total_positions = 0;
  std::size_t dont_care_positions = 0;      // We have captures
  std::size_t opponent_capture_positions = 0; // Opponent would have captures
  std::size_t real_positions = 0;           // Need actual storage
  std::size_t wins = 0;
  std::size_t losses = 0;
  std::size_t draws = 0;

  double dont_care_ratio() const {
    return total_positions > 0
      ? static_cast<double>(dont_care_positions + opponent_capture_positions) / total_positions
      : 0.0;
  }

  double compression_potential() const {
    return total_positions > 0
      ? static_cast<double>(real_positions) / total_positions
      : 1.0;
  }
};

// Analyze a tablebase and compute compression statistics
CompressionStats analyze_compression(const std::vector<Value>& tablebase, const Material& m);

// ============================================================================
// WDL Lookup with Search for Don't-Care Positions (Stage 2)
// ============================================================================

// Sentinel value for don't-care positions in compressed storage
constexpr std::uint8_t DONT_CARE = 0;  // Placeholder in compressed format

// Search performance statistics
struct SearchStats {
  std::size_t lookups = 0;           // Total lookup calls
  std::size_t direct_hits = 0;       // Positions resolved without search
  std::size_t searches = 0;          // Positions requiring search
  std::size_t total_nodes = 0;       // Total nodes visited in all searches
  std::size_t max_depth = 0;         // Maximum search depth encountered
  std::size_t terminal_wins = 0;     // Terminal positions (opponent has no pieces)
  std::size_t sub_tb_lookups = 0;    // Lookups into sub-tablebases

  double avg_nodes_per_search() const {
    return searches > 0 ? static_cast<double>(total_nodes) / searches : 0.0;
  }

  double search_ratio() const {
    return lookups > 0 ? static_cast<double>(searches) / lookups : 0.0;
  }
};

// Lookup WDL value, searching through don't-care positions if needed.
// Don't-care positions are those with captures, which are irreversible,
// so the search is naturally bounded by the number of pieces on the board.
//
// Parameters:
//   b: The board position to look up
//   tablebase: The compressed tablebase (with DONT_CARE markers)
//   m: The material configuration
//
// Returns the WDL value (WIN, LOSS, or DRAW)
Value lookup_wdl_with_search(
    const Board& b,
    const std::vector<Value>& tablebase,
    const Material& m);

// Version with sub-tablebase support and statistics tracking
// sub_tablebases: Map from material to compressed tablebase for sub-endgames
// stats: Optional output for search statistics (pass nullptr to disable)
Value lookup_wdl_with_search(
    const Board& b,
    const std::vector<Value>& tablebase,
    const Material& m,
    const std::unordered_map<Material, std::vector<Value>>& sub_tablebases,
    SearchStats* stats);

// ============================================================================
// Compressed WDL Generation (Stage 1)
// ============================================================================

// Generate a WDL tablebase with don't-care positions marked.
// Don't-care positions are marked with the DONT_CARE sentinel.
//
// Parameters:
//   original: The original (uncompressed) WDL tablebase
//   m: The material configuration
//   stats: Output statistics about the compression
//
// Returns a new vector with don't-care positions marked
std::vector<Value> mark_dont_care_positions(
    const std::vector<Value>& original,
    const Material& m,
    CompressionStats& stats);

// ============================================================================
// Block-Based Compression (Stage 3+)
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

// Compressed block header
struct CompressedBlock {
  CompressionMethod method;
  std::uint16_t compressed_size;
  std::vector<std::uint8_t> data;
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
// Block Compression/Decompression (Stage 3)
// ============================================================================

// Compress a block of values using the specified method.
// Returns the compressed data (without header).
std::vector<std::uint8_t> compress_block(
    const Value* values,
    std::size_t count,
    CompressionMethod method);

// Decompress a block of data using the specified method.
// Returns the decompressed values.
std::vector<Value> decompress_block(
    const std::uint8_t* data,
    std::size_t data_size,
    std::size_t num_values,
    CompressionMethod method);

// Try all compression methods and return the best one.
// Returns (method, compressed_data).
std::pair<CompressionMethod, std::vector<std::uint8_t>> compress_block_best(
    const Value* values,
    std::size_t count);

// Get the expected compressed size for a method (for method 0 and 1).
// Returns 0 if size cannot be determined without actually compressing.
std::size_t expected_compressed_size(std::size_t num_values, CompressionMethod method);

// ============================================================================
// CompressedTablebase Creation and Lookup (Stage 3)
// ============================================================================

// Create a compressed tablebase from a vector of values.
CompressedTablebase compress_tablebase(
    const std::vector<Value>& values,
    const Material& m);

// Look up a single value in a compressed tablebase.
// Decompresses the relevant block on demand (stateless, thread-safe).
Value lookup_compressed(
    const CompressedTablebase& tb,
    std::size_t index);

// Look up a value with search for don't-care positions.
// Combines compressed lookup with the search algorithm from Stage 2.
Value lookup_compressed_with_search(
    const Board& b,
    const CompressedTablebase& tb);

// ============================================================================
// Compression Statistics (Stage 3)
// ============================================================================

struct BlockCompressionStats {
  std::size_t total_blocks = 0;
  std::size_t method_counts[16] = {0};  // Count per method
  std::size_t uncompressed_size = 0;
  std::size_t compressed_size = 0;

  double compression_ratio() const {
    return uncompressed_size > 0
      ? static_cast<double>(uncompressed_size) / compressed_size
      : 1.0;
  }
};

// Analyze compression statistics for a compressed tablebase.
BlockCompressionStats analyze_block_compression(const CompressedTablebase& tb);

// ============================================================================
// File Format Constants (Stage 5)
// ============================================================================

constexpr char CWDL_MAGIC[4] = {'C', 'W', 'D', 'L'};
constexpr std::uint8_t CWDL_VERSION = 2;  // v2: 64-bit num_positions

// ============================================================================
// Compressed Tablebase File I/O (Stage 5)
// ============================================================================

// Save a compressed tablebase to a file.
// File format (v2):
//   [4 bytes]  Magic "CWDL"
//   [1 byte]   Version (2)
//   [6 bytes]  Material (6 piece counts)
//   [8 bytes]  num_positions (64-bit, little-endian)
//   [4 bytes]  num_blocks
//   [4 bytes × num_blocks]  Block offsets
//   [variable] Block data (concatenated compressed blocks)
//
// Returns true on success, false on error.
bool save_compressed_tablebase(const CompressedTablebase& tb, const std::string& filename);

// Load a compressed tablebase from a file.
// Returns an empty tablebase (num_positions == 0) on error.
CompressedTablebase load_compressed_tablebase(const std::string& filename);

// ============================================================================
// BitWriter/BitReader for Huffman Encoding (Stage 4)
// ============================================================================

// Bit-level writer for Huffman encoding
class BitWriter {
public:
  BitWriter();

  // Write bits (value is right-aligned, num_bits <= 32)
  void write(std::uint32_t value, int num_bits);

  // Flush remaining bits and get the result
  std::vector<std::uint8_t> finish();

  // Current bit position
  std::size_t bit_count() const { return bit_count_; }

private:
  std::vector<std::uint8_t> buffer_;
  std::uint32_t current_byte_ = 0;
  int bits_in_byte_ = 0;
  std::size_t bit_count_ = 0;
};

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
// Run-Length Statistics Collection (Stage 4)
// ============================================================================

struct RunStatistics {
  std::size_t total_blocks = 0;
  std::size_t total_runs = 0;
  std::size_t total_positions = 0;

  // Run length histogram (bucket i = lengths 2^i to 2^(i+1)-1)
  std::size_t run_length_histogram[16] = {0};

  // Prediction accuracy (run k == run k-2)
  std::size_t prediction_correct = 0;
  std::size_t prediction_total = 0;

  // Value distribution
  std::size_t value_counts[4] = {0};  // WIN, DRAW, LOSS, UNKNOWN

  // Number of distinct values per block histogram
  std::size_t distinct_value_histogram[5] = {0};  // 1, 2, 3, 4 distinct values

  double prediction_accuracy() const {
    return prediction_total > 0
      ? static_cast<double>(prediction_correct) / prediction_total
      : 0.0;
  }

  double avg_run_length() const {
    return total_runs > 0
      ? static_cast<double>(total_positions) / total_runs
      : 0.0;
  }
};

// Collect run-length statistics from a single block
void collect_block_run_statistics(const Value* values, std::size_t count, RunStatistics& stats);

// Collect statistics across all tablebases in a directory
RunStatistics collect_all_tablebase_statistics(const std::string& directory = ".");

// Print statistics summary
void print_run_statistics(const RunStatistics& stats);

// ============================================================================
// Compressed Tablebase Lookup with Search (Public API)
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

  // Get a loaded tablebase (or nullptr if not available)
  const CompressedTablebase* get_tablebase(const Material& m);

  // Clear loaded tablebases
  void clear();

  // Get the directory being used
  const std::string& directory() const { return directory_; }

private:
  std::string directory_;
  std::unordered_map<Material, CompressedTablebase> tb_cache_;

  // Load a compressed tablebase (or return cached)
  CompressedTablebase* load_or_get(const Material& m, bool warn_if_missing = true);
};

