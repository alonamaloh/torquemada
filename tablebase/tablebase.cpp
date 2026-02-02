#include "tablebase.hpp"

#ifdef WASM_BUILD
#include "../web/src/wasm/pext_polyfill.hpp"
#else
#include <x86intrin.h>
#endif

#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace {

// Flip rows in bitboard (byte swap + nibble swap within bytes)
// Different from board flip - only reverses row order
Bb flip_rows(Bb x) {
  x = __builtin_bswap32(x);
  x = (x & 0xf0f0f0f0u) >> 4 | (x & 0x0f0f0f0fu) << 4;
  return x;
}

// Memoized binomial coefficients
std::size_t choose_memo[33][33] = {};

// Combinatorial index of a bitboard (position in lexicographic ordering)
// Treats set bits as a k-combination of n elements
std::size_t index_bits(std::uint32_t bb) {
  std::size_t result = 0;
  for (int n = 0; bb; bb &= bb - 1, n++) {
    int bit_no = __builtin_ctz(bb);
    result += choose(bit_no, n + 1);
  }
  return result;
}

// Combinatorial index with mask (extract relevant bits first using pext)
std::size_t index_bits(std::uint32_t bb, std::uint32_t mask) {
  return index_bits(_pext_u32(bb, mask));
}

// Reverse of index_bits: convert index to bitboard with k bits set
std::uint32_t unindex_bits(std::size_t x, int k) {
  std::uint32_t result = 0;
  for (; k > 0; k--) {
    int n = 0;
    while (choose(n + 1, k) <= x) {
      n++;
    }
    x -= choose(n, k);
    result |= 1u << n;
  }
  return result;
}

// Reverse with mask (deposit bits back using pdep)
std::uint32_t unindex_bits(std::size_t x, int k, std::uint32_t mask) {
  return _pdep_u32(unindex_bits(x, k), mask);
}

// Masks for different board regions
constexpr Bb BACK_WHITE_MASK = 0x0000000fu;  // Row 1 (squares 0-3)
constexpr Bb BACK_BLACK_MASK = 0xf0000000u;  // Row 8 (squares 28-31)
constexpr Bb MIDDLE_MASK     = 0x0ffffff0u;  // Rows 2-7 (squares 4-27)

} // namespace

// Binomial coefficient C(n, k)
std::size_t choose(int n, int k) {
  if (k < 0 || k > n) return 0;
  if (k == 0 || k == n) return 1;
  if (choose_memo[n][k] != 0) return choose_memo[n][k];
  return choose_memo[n][k] = choose(n - 1, k - 1) + choose(n - 1, k);
}

// Extract material configuration from a board
Material get_material(const Board& b) {
  Bb wp = b.white & ~b.kings;  // White pawns
  Bb bp = b.black & ~b.kings;  // Black pawns
  Bb wk = b.white & b.kings;   // White queens
  Bb bk = b.black & b.kings;   // Black queens

  return Material{
    .back_white_pawns  = __builtin_popcount(wp & BACK_WHITE_MASK),
    .back_black_pawns  = __builtin_popcount(bp & BACK_BLACK_MASK),
    .other_white_pawns = __builtin_popcount(wp & MIDDLE_MASK),
    .other_black_pawns = __builtin_popcount(bp & MIDDLE_MASK),
    .white_queens      = __builtin_popcount(wk),
    .black_queens      = __builtin_popcount(bk),
  };
}

// Total number of positions for a material configuration
std::size_t material_size(const Material& m) {
  // Number of squares available for queens after placing pawns
  int n = 32 - m.back_white_pawns - m.back_black_pawns
            - m.other_white_pawns - m.other_black_pawns;

  std::size_t result = 1;

  // Back row pawns: 4 choose k
  result *= choose(4, m.back_white_pawns);
  result *= choose(4, m.back_black_pawns);

  // Other pawns: 24 choose k, but black pawns go in remaining squares
  result *= choose(24, m.other_white_pawns);
  result *= choose(24 - m.other_white_pawns, m.other_black_pawns);

  // Queens: placed on remaining squares
  result *= choose(n, m.white_queens);
  result *= choose(n - m.white_queens, m.black_queens);

  return result;
}

// Convert board to index
std::size_t board_to_index(const Board& b, const Material& m) {
  Bb wp = b.white & ~b.kings;
  Bb bp = b.black & ~b.kings;
  Bb wk = b.white & b.kings;
  Bb bk = b.black & b.kings;

  // Phase 1: Back white pawns (row 1, squares 0-3)
  std::size_t i0 = index_bits(wp & BACK_WHITE_MASK);

  // Phase 2: Back black pawns (row 8, squares 28-31)
  std::size_t i1 = index_bits(bp & BACK_BLACK_MASK, BACK_BLACK_MASK);

  // Phase 3: Other white pawns (rows 2-7)
  // Note: flip_rows because indexing expects low bits first
  std::size_t i2 = index_bits(flip_rows(wp) & MIDDLE_MASK, MIDDLE_MASK);

  // Phase 4: Other black pawns in remaining middle squares
  std::size_t i3 = index_bits(bp & MIDDLE_MASK, ~wp & MIDDLE_MASK);

  // Phase 5: White queens in remaining squares
  std::size_t i4 = index_bits(wk, ~(wp | bp));

  // Phase 6: Black queens in remaining squares
  std::size_t i5 = index_bits(bk, ~(wp | bp | wk));

  // Combine indices using mixed-radix representation
  int n = 32 - m.back_white_pawns - m.back_black_pawns
            - m.other_white_pawns - m.other_black_pawns;

  std::size_t result = 0;
  std::size_t multiplier = 1;

  result += i5 * multiplier;
  multiplier *= choose(n - m.white_queens, m.black_queens);

  result += i4 * multiplier;
  multiplier *= choose(n, m.white_queens);

  result += i3 * multiplier;
  multiplier *= choose(24 - m.other_white_pawns, m.other_black_pawns);

  result += i2 * multiplier;
  multiplier *= choose(24, m.other_white_pawns);

  result += i1 * multiplier;
  multiplier *= choose(4, m.back_black_pawns);

  result += i0 * multiplier;

  return result;
}

// Convert index back to board
Board index_to_board(std::size_t idx, const Material& m) {
  int n = 32 - m.back_white_pawns - m.back_black_pawns
            - m.other_white_pawns - m.other_black_pawns;

  // Extract indices in reverse order
  std::size_t n_bk = choose(n - m.white_queens, m.black_queens);
  std::size_t i5 = idx % n_bk;
  idx /= n_bk;

  std::size_t n_wk = choose(n, m.white_queens);
  std::size_t i4 = idx % n_wk;
  idx /= n_wk;

  std::size_t n_obp = choose(24 - m.other_white_pawns, m.other_black_pawns);
  std::size_t i3 = idx % n_obp;
  idx /= n_obp;

  std::size_t n_owp = choose(24, m.other_white_pawns);
  std::size_t i2 = idx % n_owp;
  idx /= n_owp;

  std::size_t n_bbp = choose(4, m.back_black_pawns);
  std::size_t i1 = idx % n_bbp;
  idx /= n_bbp;

  std::size_t i0 = idx;  // Remaining is back white pawns index

  // Reconstruct bitboards
  Bb wp = unindex_bits(i0, m.back_white_pawns);
  Bb bp = unindex_bits(i1, m.back_black_pawns, BACK_BLACK_MASK);
  wp |= flip_rows(unindex_bits(i2, m.other_white_pawns, MIDDLE_MASK));
  bp |= unindex_bits(i3, m.other_black_pawns, ~wp & MIDDLE_MASK);
  Bb wk = unindex_bits(i4, m.white_queens, ~(wp | bp));
  Bb bk = unindex_bits(i5, m.black_queens, ~(wp | bp | wk));

  Board b;
  b.white = wp | wk;
  b.black = bp | bk;
  b.kings = wk | bk;
  b.n_reversible = 0;
  return b;
}

// Filename for a material configuration
std::string tablebase_filename(const Material& m) {
  std::ostringstream oss;
  oss << "tb_" << m.back_white_pawns << m.back_black_pawns
      << m.other_white_pawns << m.other_black_pawns
      << m.white_queens << m.black_queens << ".bin";
  return oss.str();
}

// Filename for a compressed WDL tablebase
std::string compressed_tablebase_filename(const Material& m) {
  std::ostringstream oss;
  oss << "cwdl_" << m.back_white_pawns << m.back_black_pawns
      << m.other_white_pawns << m.other_black_pawns
      << m.white_queens << m.black_queens << ".bin";
  return oss.str();
}

// Save tablebase to file
void save_tablebase(const std::vector<Value>& table, const Material& m) {
  std::string filename = tablebase_filename(m);
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Write header: material config and size
  file.write(reinterpret_cast<const char*>(&m), sizeof(Material));
  std::size_t size = table.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));

  // Pack values: 4 values per byte (2 bits each)
  std::vector<std::uint8_t> packed((size + 3) / 4);
  for (std::size_t i = 0; i < size; ++i) {
    int byte_idx = i / 4;
    int bit_offset = (i % 4) * 2;
    packed[byte_idx] |= static_cast<std::uint8_t>(table[i]) << bit_offset;
  }

  file.write(reinterpret_cast<const char*>(packed.data()), packed.size());
}

// Load tablebase from file
// Throws std::runtime_error if file cannot be opened or read
std::vector<Value> load_tablebase(const Material& m) {
  std::string filename = tablebase_filename(m);
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open tablebase file: " + filename);
  }

  // Read and verify header
  Material stored_m;
  file.read(reinterpret_cast<char*>(&stored_m), sizeof(Material));
  if (!file) {
    throw std::runtime_error("Failed to read material from tablebase file: " + filename);
  }
  if (!(stored_m == m)) {
    throw std::runtime_error("Material mismatch in tablebase file: " + filename);
  }

  std::size_t size;
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  if (!file) {
    throw std::runtime_error("Failed to read size from tablebase file: " + filename);
  }

  // Read packed data
  std::vector<std::uint8_t> packed((size + 3) / 4);
  file.read(reinterpret_cast<char*>(packed.data()), packed.size());
  if (!file) {
    throw std::runtime_error("Failed to read data from tablebase file: " + filename);
  }

  // Unpack values
  std::vector<Value> table(size);
  for (std::size_t i = 0; i < size; ++i) {
    int byte_idx = i / 4;
    int bit_offset = (i % 4) * 2;
    table[i] = static_cast<Value>((packed[byte_idx] >> bit_offset) & 0x3);
  }

  return table;
}

// Check if tablebase exists
bool tablebase_exists(const Material& m) {
  return std::filesystem::exists(tablebase_filename(m));
}

// Filename for DTM tablebase
std::string dtm_filename(const Material& m) {
  std::ostringstream oss;
  oss << "dtm_" << m.back_white_pawns << m.back_black_pawns
      << m.other_white_pawns << m.other_black_pawns
      << m.white_queens << m.black_queens << ".bin";
  return oss.str();
}

// Save DTM tablebase to file (1 byte per position)
void save_dtm(const std::vector<DTM>& table, const Material& m) {
  std::string filename = dtm_filename(m);
  std::ofstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }

  // Write header: version, material config, and size
  std::uint8_t version = DTM_FORMAT_VERSION;
  file.write(reinterpret_cast<const char*>(&version), sizeof(version));
  file.write(reinterpret_cast<const char*>(&m), sizeof(Material));
  std::size_t size = table.size();
  file.write(reinterpret_cast<const char*>(&size), sizeof(size));

  // Store lower byte directly (DTM values fit in int8_t range)
  std::vector<std::int8_t> packed(size);
  for (std::size_t i = 0; i < size; ++i) {
    packed[i] = static_cast<std::int8_t>(table[i]);
  }
  file.write(reinterpret_cast<const char*>(packed.data()), packed.size());
}

// Load DTM tablebase from file
// Throws std::runtime_error if file cannot be opened or read
std::vector<DTM> load_dtm(const Material& m) {
  std::string filename = dtm_filename(m);
  std::ifstream file(filename, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open DTM file: " + filename);
  }

  // Read and verify header
  std::uint8_t version;
  file.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (!file) {
    throw std::runtime_error("Failed to read version from DTM file: " + filename);
  }

  Material stored_m;
  file.read(reinterpret_cast<char*>(&stored_m), sizeof(Material));
  if (!file) {
    throw std::runtime_error("Failed to read material from DTM file: " + filename);
  }
  if (!(stored_m == m)) {
    throw std::runtime_error("Material mismatch in DTM file: " + filename);
  }

  std::size_t size;
  file.read(reinterpret_cast<char*>(&size), sizeof(size));
  if (!file) {
    throw std::runtime_error("Failed to read size from DTM file: " + filename);
  }

  // Read 1-byte values and sign-extend to DTM
  std::vector<std::int8_t> packed(size);
  file.read(reinterpret_cast<char*>(packed.data()), packed.size());
  if (!file) {
    throw std::runtime_error("Failed to read data from DTM file: " + filename);
  }

  std::vector<DTM> table(size);
  for (std::size_t i = 0; i < size; ++i) {
    table[i] = static_cast<DTM>(packed[i]);  // Sign extension
  }

  return table;
}

// Check if DTM tablebase exists
bool dtm_exists(const Material& m) {
  return std::filesystem::exists(dtm_filename(m));
}
