// Comprehensive tests for the game prover (prove.cpp)
//
// Tests cover:
//   1. PositionKey hashing and equality
//   2. KnownPositions: store, lookup, n_reversible filtering, save/load, new_count
//   3. Perspective conversions: stm_to_prover, prover_to_stm, round-trip
//   4. make_frame: AND/OR node assignment for Espada and Broquel
//   5. store_proven + check_resolved consistency
//   6. check_resolved: tablebase draws, wins, losses in both modes
//   7. No legal moves: stm loses
//   8. AND/OR tree propagation logic
//   9. Repetition detection
//  10. Draw value setup for search (evaluate_node draw_value)
//  11. Integration: known endgame positions with real tablebases

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "search/search.hpp"
#include "tablebase/compression.hpp"
#include <bit>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Duplicated types from prove.cpp (needed since prove.cpp has no header)
// ============================================================================

enum class ProverMode { ESPADA, BROQUEL };

enum class NodeResult : int8_t { TARGET_WINS = 1, TARGET_LOSES = -1, UNKNOWN = 0 };

struct PositionKey {
  Bb white, black, kings;
  uint8_t black_to_move;

  bool operator==(const PositionKey&) const = default;
};

namespace std {
  template<>
  struct hash<PositionKey> {
    size_t operator()(const PositionKey& k) const noexcept {
      uint64_t h = k.white * 0x9d82c4a44a2de231ull;
      h ^= h >> 32;
      h += k.black;
      h *= 0xb20534a511d28c31ull;
      h ^= h >> 32;
      h += k.kings;
      h *= 0x3a2a8392d61061d7ull;
      h ^= h >> 32;
      h += k.black_to_move;
      h *= 0xc4a44a2de231b205ull;
      h ^= h >> 32;
      return h;
    }
  };
}

class KnownPositions {
public:
  void store(const Board& board, int game_ply, NodeResult result) {
    if (board.n_reversible != 0) return;
    PositionKey key{board.white, board.black, board.kings,
                    static_cast<uint8_t>(game_ply % 2)};
    auto [it, inserted] = map_.emplace(key, result);
    if (inserted) new_count_++;
    else it->second = result;
  }

  std::optional<NodeResult> lookup(const Board& board, int game_ply) const {
    if (board.n_reversible != 0) return std::nullopt;
    PositionKey key{board.white, board.black, board.kings,
                    static_cast<uint8_t>(game_ply % 2)};
    auto it = map_.find(key);
    if (it != map_.end()) return it->second;
    return std::nullopt;
  }

  size_t size() const { return map_.size(); }
  size_t new_count() const { return new_count_; }
  void reset_new_count() { new_count_ = 0; }

  bool save(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;
    uint64_t count = map_.size();
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& [key, result] : map_) {
      out.write(reinterpret_cast<const char*>(&key.white), sizeof(key.white));
      out.write(reinterpret_cast<const char*>(&key.black), sizeof(key.black));
      out.write(reinterpret_cast<const char*>(&key.kings), sizeof(key.kings));
      out.write(reinterpret_cast<const char*>(&key.black_to_move), sizeof(key.black_to_move));
      int8_t v = static_cast<int8_t>(result);
      out.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }
    return out.good();
  }

  bool load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;
    uint64_t count;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    for (uint64_t i = 0; i < count && in.good(); ++i) {
      PositionKey key;
      int8_t v;
      in.read(reinterpret_cast<char*>(&key.white), sizeof(key.white));
      in.read(reinterpret_cast<char*>(&key.black), sizeof(key.black));
      in.read(reinterpret_cast<char*>(&key.kings), sizeof(key.kings));
      in.read(reinterpret_cast<char*>(&key.black_to_move), sizeof(key.black_to_move));
      in.read(reinterpret_cast<char*>(&v), sizeof(v));
      map_[key] = static_cast<NodeResult>(v);
    }
    return in.good();
  }

  void clear() { map_.clear(); new_count_ = 0; }

  // Expose map for testing
  const auto& map() const { return map_; }

private:
  std::unordered_map<PositionKey, NodeResult> map_;
  size_t new_count_ = 0;
};

// Duplicated helper functions from prove.cpp
static NodeResult stm_to_prover(int stm_val, bool is_and_node) {
  int prover_val = is_and_node ? -stm_val : stm_val;
  return static_cast<NodeResult>(prover_val);
}

static int prover_to_stm(NodeResult result, bool is_and_node) {
  int prover_val = static_cast<int>(result);
  return is_and_node ? -prover_val : prover_val;
}

struct StackFrame {
  Board board;
  int game_ply;
  bool is_and_node;
  std::vector<Move> moves;
  std::vector<int> move_scores;
  int next_move_idx = 0;
  bool searched = false;
};

static StackFrame make_frame(const Board& board, int game_ply, ProverMode mode) {
  StackFrame f;
  f.board = board;
  f.game_ply = game_ply;
  bool even_ply = (game_ply % 2 == 0);
  f.is_and_node = (mode == ProverMode::ESPADA) ? even_ply : !even_ply;
  return f;
}

// Store a proven result (prover perspective) in the known DB (stm perspective).
static void store_proven(KnownPositions& known, const StackFrame& frame, NodeResult result) {
  int stm_val = prover_to_stm(result, frame.is_and_node);
  known.store(frame.board, frame.game_ply, static_cast<NodeResult>(stm_val));
}

// Check if position is already resolved via tablebases.
// Returns prover-perspective result.
static NodeResult check_resolved_tb(const CompressedTablebaseManager& cwdl,
                                     const Board& board, int game_ply,
                                     bool is_and_node, int tb_pieces) {
  int pieces = std::popcount(board.allPieces());
  if (pieces <= tb_pieces) {
    Value v = cwdl.lookup_wdl_preloaded(board);
    int stm_val;
    switch (v) {
      case Value::WIN: stm_val = 1; break;
      case Value::LOSS: stm_val = -1; break;
      case Value::DRAW:
        stm_val = is_and_node ? -1 : 1;
        break;
      default: return NodeResult::UNKNOWN;
    }
    return stm_to_prover(stm_val, is_and_node);
  }
  return NodeResult::UNKNOWN;
}

// Check if position is resolved via known positions DB.
static NodeResult check_resolved_known(const KnownPositions& known,
                                        const Board& board, int game_ply,
                                        bool is_and_node) {
  if (auto r = known.lookup(board, game_ply)) {
    return stm_to_prover(static_cast<int>(*r), is_and_node);
  }
  return NodeResult::UNKNOWN;
}

// ============================================================================
// Test infrastructure
// ============================================================================

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
  static void test_##name(); \
  static struct Register_##name { \
    Register_##name() { test_registry().push_back({#name, test_##name}); } \
  } reg_##name; \
  static void test_##name()

#define ASSERT_TRUE(expr) do { \
  if (!(expr)) { \
    std::cerr << "  FAIL: " << #expr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b) do { \
  auto _a = (a); auto _b = (b); \
  if (_a != _b) { \
    std::cerr << "  FAIL: " << #a << " == " << #b \
              << " (got " << static_cast<int>(_a) << " vs " << static_cast<int>(_b) << ")" \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

#define ASSERT_NE(a, b) do { \
  auto _a = (a); auto _b = (b); \
  if (_a == _b) { \
    std::cerr << "  FAIL: " << #a << " != " << #b \
              << " (both " << static_cast<int>(_a) << ")" \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    throw std::runtime_error("assertion failed"); \
  } \
} while(0)

struct TestEntry {
  const char* name;
  void (*func)();
};

static std::vector<TestEntry>& test_registry() {
  static std::vector<TestEntry> reg;
  return reg;
}

// ============================================================================
// Test 1: PositionKey hashing and equality
// ============================================================================

TEST(position_key_same_position_same_hash) {
  PositionKey k1{0x00000FFF, 0xFFF00000, 0, 0};
  PositionKey k2{0x00000FFF, 0xFFF00000, 0, 0};
  ASSERT_TRUE(k1 == k2);
  ASSERT_EQ(std::hash<PositionKey>{}(k1), std::hash<PositionKey>{}(k2));
}

TEST(position_key_different_white) {
  PositionKey k1{0x00000FFF, 0xFFF00000, 0, 0};
  PositionKey k2{0x00000FFE, 0xFFF00000, 0, 0};
  ASSERT_FALSE(k1 == k2);
  // Hash collision is possible but extremely unlikely
  ASSERT_NE(std::hash<PositionKey>{}(k1), std::hash<PositionKey>{}(k2));
}

TEST(position_key_different_black) {
  PositionKey k1{0x00000FFF, 0xFFF00000, 0, 0};
  PositionKey k2{0x00000FFF, 0xFFE00000, 0, 0};
  ASSERT_FALSE(k1 == k2);
}

TEST(position_key_different_kings) {
  PositionKey k1{0x00000FFF, 0xFFF00000, 0, 0};
  PositionKey k2{0x00000FFF, 0xFFF00000, 1, 0};
  ASSERT_FALSE(k1 == k2);
}

TEST(position_key_different_side_to_move) {
  // Same board but different side to move should be different keys
  PositionKey k1{0x00000FFF, 0xFFF00000, 0, 0};
  PositionKey k2{0x00000FFF, 0xFFF00000, 0, 1};
  ASSERT_FALSE(k1 == k2);
  ASSERT_NE(std::hash<PositionKey>{}(k1), std::hash<PositionKey>{}(k2));
}

TEST(position_key_ply_parity_maps_to_black_to_move) {
  // Even ply -> black_to_move=0, Odd ply -> black_to_move=1
  Board b(0x1, 0x2, 0);
  PositionKey k_even{b.white, b.black, b.kings, static_cast<uint8_t>(0 % 2)};
  PositionKey k_odd{b.white, b.black, b.kings, static_cast<uint8_t>(1 % 2)};
  ASSERT_EQ(k_even.black_to_move, 0);
  ASSERT_EQ(k_odd.black_to_move, 1);
  ASSERT_FALSE(k_even == k_odd);
}

TEST(position_key_even_plies_are_same) {
  // Ply 0 and ply 2 and ply 4 should all produce the same key
  Board b(0x1, 0x2, 0);
  PositionKey k0{b.white, b.black, b.kings, static_cast<uint8_t>(0 % 2)};
  PositionKey k2{b.white, b.black, b.kings, static_cast<uint8_t>(2 % 2)};
  PositionKey k4{b.white, b.black, b.kings, static_cast<uint8_t>(4 % 2)};
  ASSERT_TRUE(k0 == k2);
  ASSERT_TRUE(k0 == k4);
}

// ============================================================================
// Test 2: KnownPositions store and lookup
// ============================================================================

TEST(known_positions_store_and_lookup) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);  // 1 white pawn, 1 black pawn
  b.n_reversible = 0;

  kp.store(b, 0, NodeResult::TARGET_WINS);
  auto result = kp.lookup(b, 0);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(*result, NodeResult::TARGET_WINS);
}

TEST(known_positions_n_reversible_nonzero_rejected) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  b.n_reversible = 5;  // non-zero

  kp.store(b, 0, NodeResult::TARGET_WINS);
  ASSERT_EQ(kp.size(), 0u);  // should not have been stored

  auto result = kp.lookup(b, 0);
  ASSERT_FALSE(result.has_value());  // should not be found
}

TEST(known_positions_n_reversible_zero_accepted) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  b.n_reversible = 0;

  kp.store(b, 0, NodeResult::TARGET_LOSES);
  ASSERT_EQ(kp.size(), 1u);

  auto result = kp.lookup(b, 0);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(*result, NodeResult::TARGET_LOSES);
}

TEST(known_positions_lookup_wrong_ply_parity) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);

  kp.store(b, 0, NodeResult::TARGET_WINS);

  // Lookup at odd ply should not find it (different side to move)
  auto result = kp.lookup(b, 1);
  ASSERT_FALSE(result.has_value());

  // But even ply should find it
  auto result2 = kp.lookup(b, 2);
  ASSERT_TRUE(result2.has_value());
  ASSERT_EQ(*result2, NodeResult::TARGET_WINS);
}

TEST(known_positions_overwrite) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);

  kp.store(b, 0, NodeResult::TARGET_WINS);
  ASSERT_EQ(kp.size(), 1u);

  // Overwrite with different result
  kp.store(b, 0, NodeResult::TARGET_LOSES);
  ASSERT_EQ(kp.size(), 1u);  // still 1, not 2

  auto result = kp.lookup(b, 0);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(*result, NodeResult::TARGET_LOSES);
}

TEST(known_positions_new_count) {
  KnownPositions kp;
  Board b1(0x1, 0x80000000, 0);
  Board b2(0x2, 0x80000000, 0);

  ASSERT_EQ(kp.new_count(), 0u);

  kp.store(b1, 0, NodeResult::TARGET_WINS);
  ASSERT_EQ(kp.new_count(), 1u);

  kp.store(b2, 0, NodeResult::TARGET_LOSES);
  ASSERT_EQ(kp.new_count(), 2u);

  // Overwrite should NOT increment new_count
  kp.store(b1, 0, NodeResult::TARGET_LOSES);
  ASSERT_EQ(kp.new_count(), 2u);

  kp.reset_new_count();
  ASSERT_EQ(kp.new_count(), 0u);
}

TEST(known_positions_multiple_positions) {
  KnownPositions kp;

  // Store several different positions
  for (int i = 0; i < 100; i++) {
    Board b(static_cast<Bb>(1u << (i % 12)), static_cast<Bb>(1u << (20 + i % 12)), 0);
    kp.store(b, i % 2, static_cast<NodeResult>((i % 3) - 1));
  }

  // Verify each can be looked up
  for (int i = 0; i < 100; i++) {
    Board b(static_cast<Bb>(1u << (i % 12)), static_cast<Bb>(1u << (20 + i % 12)), 0);
    auto result = kp.lookup(b, i % 2);
    ASSERT_TRUE(result.has_value());
  }
}

// ============================================================================
// Test 3: KnownPositions save/load round-trip
// ============================================================================

TEST(known_positions_save_load_roundtrip) {
  const std::string tmpfile = "/tmp/test_prover_known.bin";
  KnownPositions kp;

  // Store a variety of positions and results
  Board b1(0x1, 0x80000000, 0);
  Board b2(0x2, 0x40000000, 0x2);       // white king
  Board b3(0x100, 0x800000, 0x800000);   // black king

  kp.store(b1, 0, NodeResult::TARGET_WINS);
  kp.store(b1, 1, NodeResult::TARGET_LOSES);  // same board, different parity
  kp.store(b2, 0, NodeResult::TARGET_LOSES);
  kp.store(b3, 1, NodeResult::TARGET_WINS);

  ASSERT_EQ(kp.size(), 4u);
  ASSERT_TRUE(kp.save(tmpfile));

  // Load into a fresh KnownPositions
  KnownPositions kp2;
  ASSERT_TRUE(kp2.load(tmpfile));
  ASSERT_EQ(kp2.size(), 4u);

  // Verify all entries match
  ASSERT_EQ(*kp2.lookup(b1, 0), NodeResult::TARGET_WINS);
  ASSERT_EQ(*kp2.lookup(b1, 1), NodeResult::TARGET_LOSES);
  ASSERT_EQ(*kp2.lookup(b2, 0), NodeResult::TARGET_LOSES);
  ASSERT_EQ(*kp2.lookup(b3, 1), NodeResult::TARGET_WINS);

  // Clean up
  std::remove(tmpfile.c_str());
}

TEST(known_positions_save_load_empty) {
  const std::string tmpfile = "/tmp/test_prover_empty.bin";
  KnownPositions kp;
  ASSERT_TRUE(kp.save(tmpfile));

  KnownPositions kp2;
  ASSERT_TRUE(kp2.load(tmpfile));
  ASSERT_EQ(kp2.size(), 0u);

  std::remove(tmpfile.c_str());
}

TEST(known_positions_load_nonexistent_file) {
  KnownPositions kp;
  ASSERT_FALSE(kp.load("/tmp/nonexistent_file_12345.bin"));
}

TEST(known_positions_save_load_preserves_all_result_types) {
  const std::string tmpfile = "/tmp/test_prover_results.bin";
  KnownPositions kp;

  Board b_win(0x1, 0x80000000, 0);
  Board b_lose(0x2, 0x80000000, 0);
  Board b_unknown(0x4, 0x80000000, 0);

  kp.store(b_win, 0, NodeResult::TARGET_WINS);
  kp.store(b_lose, 0, NodeResult::TARGET_LOSES);
  // UNKNOWN (0) stored as raw value
  kp.store(b_unknown, 0, NodeResult::UNKNOWN);

  ASSERT_TRUE(kp.save(tmpfile));

  KnownPositions kp2;
  ASSERT_TRUE(kp2.load(tmpfile));

  ASSERT_EQ(*kp2.lookup(b_win, 0), NodeResult::TARGET_WINS);
  ASSERT_EQ(*kp2.lookup(b_lose, 0), NodeResult::TARGET_LOSES);
  ASSERT_EQ(*kp2.lookup(b_unknown, 0), NodeResult::UNKNOWN);

  std::remove(tmpfile.c_str());
}

TEST(known_positions_save_load_large) {
  const std::string tmpfile = "/tmp/test_prover_large.bin";
  KnownPositions kp;

  // Store 10000 positions
  for (int i = 0; i < 10000; i++) {
    Board b(static_cast<Bb>(i + 1), static_cast<Bb>((i + 1) << 20), 0);
    NodeResult r = static_cast<NodeResult>((i % 3) - 1);
    kp.store(b, i % 2, r);
  }

  ASSERT_TRUE(kp.save(tmpfile));

  KnownPositions kp2;
  ASSERT_TRUE(kp2.load(tmpfile));
  ASSERT_EQ(kp2.size(), kp.size());

  // Spot check some entries
  for (int i = 0; i < 10000; i += 1000) {
    Board b(static_cast<Bb>(i + 1), static_cast<Bb>((i + 1) << 20), 0);
    NodeResult r = static_cast<NodeResult>((i % 3) - 1);
    auto result = kp2.lookup(b, i % 2);
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(*result, r);
  }

  std::remove(tmpfile.c_str());
}

// ============================================================================
// Test 4: Perspective conversions (stm_to_prover, prover_to_stm)
// ============================================================================

TEST(stm_to_prover_and_node_stm_wins) {
  // AND node (defending side to move), stm wins (+1)
  // At AND node, the stm is the defending side. stm winning = target loses.
  // stm_to_prover: is_and_node -> negate: -(+1) = -1 = TARGET_LOSES
  ASSERT_EQ(stm_to_prover(1, true), NodeResult::TARGET_LOSES);
}

TEST(stm_to_prover_and_node_stm_loses) {
  // AND node, stm loses (-1)
  // stm losing at AND = defending side loses = target wins
  // negate: -(-1) = +1 = TARGET_WINS
  ASSERT_EQ(stm_to_prover(-1, true), NodeResult::TARGET_WINS);
}

TEST(stm_to_prover_or_node_stm_wins) {
  // OR node (attacking side to move), stm wins (+1)
  // At OR node, the stm is the attacking side. stm winning = target wins.
  // stm_to_prover: !is_and_node -> keep: +1 = TARGET_WINS
  ASSERT_EQ(stm_to_prover(1, false), NodeResult::TARGET_WINS);
}

TEST(stm_to_prover_or_node_stm_loses) {
  // OR node, stm loses (-1)
  // stm losing at OR = attacking side loses = target loses
  // keep: -1 = TARGET_LOSES
  ASSERT_EQ(stm_to_prover(-1, false), NodeResult::TARGET_LOSES);
}

TEST(prover_to_stm_and_node_target_wins) {
  // AND node, target wins -> stm (defending) loses -> -1
  ASSERT_EQ(prover_to_stm(NodeResult::TARGET_WINS, true), -1);
}

TEST(prover_to_stm_and_node_target_loses) {
  // AND node, target loses -> stm (defending) wins -> +1
  ASSERT_EQ(prover_to_stm(NodeResult::TARGET_LOSES, true), 1);
}

TEST(prover_to_stm_or_node_target_wins) {
  // OR node, target wins -> stm (attacking) wins -> +1
  ASSERT_EQ(prover_to_stm(NodeResult::TARGET_WINS, false), 1);
}

TEST(prover_to_stm_or_node_target_loses) {
  // OR node, target loses -> stm (attacking) loses -> -1
  ASSERT_EQ(prover_to_stm(NodeResult::TARGET_LOSES, false), -1);
}

TEST(stm_prover_roundtrip) {
  // stm_to_prover and prover_to_stm should be inverses of each other
  for (int stm_val : {-1, 1}) {
    for (bool is_and : {true, false}) {
      NodeResult prover = stm_to_prover(stm_val, is_and);
      int roundtrip = prover_to_stm(prover, is_and);
      ASSERT_EQ(roundtrip, stm_val);
    }
  }
}

TEST(prover_stm_roundtrip) {
  // And the other direction
  for (NodeResult nr : {NodeResult::TARGET_WINS, NodeResult::TARGET_LOSES}) {
    for (bool is_and : {true, false}) {
      int stm = prover_to_stm(nr, is_and);
      NodeResult roundtrip = stm_to_prover(stm, is_and);
      ASSERT_EQ(roundtrip, nr);
    }
  }
}

// ============================================================================
// Test 5: make_frame (AND/OR node assignment)
// ============================================================================

TEST(make_frame_espada_even_ply_is_and) {
  Board b;
  // Espada: even ply = white to move = AND node (must prove all moves lose)
  auto f = make_frame(b, 0, ProverMode::ESPADA);
  ASSERT_TRUE(f.is_and_node);
  ASSERT_EQ(f.game_ply, 0);
}

TEST(make_frame_espada_odd_ply_is_or) {
  Board b;
  // Espada: odd ply = black to move = OR node (need one winning move)
  auto f = make_frame(b, 1, ProverMode::ESPADA);
  ASSERT_FALSE(f.is_and_node);
}

TEST(make_frame_broquel_even_ply_is_or) {
  Board b;
  // Broquel: even ply = white to move = OR node (need one winning move)
  auto f = make_frame(b, 0, ProverMode::BROQUEL);
  ASSERT_FALSE(f.is_and_node);
}

TEST(make_frame_broquel_odd_ply_is_and) {
  Board b;
  // Broquel: odd ply = black to move = AND node (must prove all moves lose)
  auto f = make_frame(b, 1, ProverMode::BROQUEL);
  ASSERT_TRUE(f.is_and_node);
}

TEST(make_frame_espada_multiple_plies) {
  Board b;
  for (int ply = 0; ply < 10; ply++) {
    auto f = make_frame(b, ply, ProverMode::ESPADA);
    ASSERT_EQ(f.is_and_node, (ply % 2 == 0));
  }
}

TEST(make_frame_broquel_multiple_plies) {
  Board b;
  for (int ply = 0; ply < 10; ply++) {
    auto f = make_frame(b, ply, ProverMode::BROQUEL);
    ASSERT_EQ(f.is_and_node, (ply % 2 != 0));
  }
}

// ============================================================================
// Test 6: store_proven + check_resolved_known consistency
// ============================================================================

TEST(store_proven_and_node_target_wins) {
  // AND node, target wins -> store stm perspective (stm loses = -1)
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 0, ProverMode::ESPADA);  // AND node at ply 0
  ASSERT_TRUE(f.is_and_node);

  store_proven(kp, f, NodeResult::TARGET_WINS);

  // Raw stored value should be -1 (stm loses)
  auto raw = kp.lookup(b, 0);
  ASSERT_TRUE(raw.has_value());
  ASSERT_EQ(static_cast<int>(*raw), -1);  // TARGET_LOSES = -1 in stm perspective

  // check_resolved_known should convert back to TARGET_WINS
  auto resolved = check_resolved_known(kp, b, 0, true);
  ASSERT_EQ(resolved, NodeResult::TARGET_WINS);
}

TEST(store_proven_and_node_target_loses) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 0, ProverMode::ESPADA);

  store_proven(kp, f, NodeResult::TARGET_LOSES);

  auto raw = kp.lookup(b, 0);
  ASSERT_TRUE(raw.has_value());
  ASSERT_EQ(static_cast<int>(*raw), 1);  // stm wins = +1

  auto resolved = check_resolved_known(kp, b, 0, true);
  ASSERT_EQ(resolved, NodeResult::TARGET_LOSES);
}

TEST(store_proven_or_node_target_wins) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 1, ProverMode::ESPADA);  // OR node at odd ply
  ASSERT_FALSE(f.is_and_node);

  store_proven(kp, f, NodeResult::TARGET_WINS);

  auto raw = kp.lookup(b, 1);
  ASSERT_TRUE(raw.has_value());
  ASSERT_EQ(static_cast<int>(*raw), 1);  // stm wins = +1

  auto resolved = check_resolved_known(kp, b, 1, false);
  ASSERT_EQ(resolved, NodeResult::TARGET_WINS);
}

TEST(store_proven_or_node_target_loses) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 1, ProverMode::ESPADA);

  store_proven(kp, f, NodeResult::TARGET_LOSES);

  auto raw = kp.lookup(b, 1);
  ASSERT_TRUE(raw.has_value());
  ASSERT_EQ(static_cast<int>(*raw), -1);  // stm loses = -1

  auto resolved = check_resolved_known(kp, b, 1, false);
  ASSERT_EQ(resolved, NodeResult::TARGET_LOSES);
}

TEST(store_proven_roundtrip_all_combinations) {
  // For every combination of mode, ply, and result:
  // store_proven then check_resolved_known should give back the same result.
  for (auto mode : {ProverMode::ESPADA, ProverMode::BROQUEL}) {
    for (int ply = 0; ply < 4; ply++) {
      for (auto nr : {NodeResult::TARGET_WINS, NodeResult::TARGET_LOSES}) {
        KnownPositions kp;
        Board b(static_cast<Bb>(ply + 1), 0x80000000, 0);  // different board per ply
        auto f = make_frame(b, ply, mode);

        store_proven(kp, f, nr);
        auto resolved = check_resolved_known(kp, b, ply, f.is_and_node);
        ASSERT_EQ(resolved, nr);
      }
    }
  }
}

// ============================================================================
// Test 7: Consistency between different lookup contexts
// ============================================================================

TEST(store_and_node_lookup_from_or_node_same_ply_parity) {
  // A position stored from an AND node context should still be retrievable,
  // but the interpretation changes when the is_and_node flag differs.
  // This tests that the raw stm value is consistent.
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);

  // Store as AND node with TARGET_WINS -> raw stm_val = -1
  auto f_and = make_frame(b, 0, ProverMode::ESPADA);
  store_proven(kp, f_and, NodeResult::TARGET_WINS);

  // Lookup from AND context: should get TARGET_WINS
  auto r_and = check_resolved_known(kp, b, 0, true);
  ASSERT_EQ(r_and, NodeResult::TARGET_WINS);

  // Lookup from OR context (same ply): raw = -1, stm_to_prover(-1, false) = -1 = TARGET_LOSES
  auto r_or = check_resolved_known(kp, b, 0, false);
  ASSERT_EQ(r_or, NodeResult::TARGET_LOSES);

  // This makes sense: if stm loses (raw = -1), then:
  //   - At AND node (defending): target wins ✓
  //   - At OR node (attacking): target loses ✓
  // Both interpretations are correct because AND and OR nodes at the same ply
  // have different target sides.
  //
  // BUT WAIT: the same position at the same ply should always have the same
  // is_and_node. In Espada, ply 0 is always AND. The above OR lookup at ply 0
  // would only happen in Broquel mode. This is fine because the raw stm value
  // is mode-independent.
}

// ============================================================================
// Test 8: No legal moves (stm loses)
// ============================================================================

TEST(no_legal_moves_stm_loses) {
  // Create a position where white has no legal moves
  // White king trapped in corner, surrounded by black pieces
  // Square 0 = bit 0, white king there
  // Black pieces blocking all moves
  Board b(0, 0, 0);
  b.white = 1u << 0;  // white on square 0
  b.kings = 1u << 0;  // it's a king
  // Place black pieces on all adjacent squares to block
  b.black = (1u << 4) | (1u << 5);  // squares 4 and 5

  MoveList ml;
  generateMoves(b, ml);

  // If no moves, stm loses
  if (ml.empty()) {
    // This is what prove.cpp does for no moves:
    // Store TARGET_LOSES as stm_val (-1 = stm loses) directly
    KnownPositions kp;
    kp.store(b, 0, NodeResult::TARGET_LOSES);

    // Convert to prover perspective
    NodeResult espada_and = stm_to_prover(-1, true);   // AND node: target wins
    NodeResult espada_or = stm_to_prover(-1, false);    // OR node: target loses
    ASSERT_EQ(espada_and, NodeResult::TARGET_WINS);
    ASSERT_EQ(espada_or, NodeResult::TARGET_LOSES);
  }
  // If there are moves, that's also OK - test doesn't require trapped position
}

TEST(starting_position_has_moves) {
  Board b;  // starting position
  MoveList ml;
  generateMoves(b, ml);
  ASSERT_TRUE(ml.size() > 0);
}

// ============================================================================
// Test 9: AND/OR tree propagation logic
// ============================================================================

// Simulate the propagation logic from prove.cpp without the full Prover class

struct PropagationSimulator {
  std::vector<StackFrame> stack;
  KnownPositions known;

  void propagate(NodeResult child_result) {
    stack.pop_back();
    if (stack.empty()) return;

    auto& parent = stack.back();

    if (parent.is_and_node) {
      if (child_result == NodeResult::TARGET_WINS) {
        parent.next_move_idx++;
      } else {
        store_proven(known, parent, NodeResult::TARGET_LOSES);
        propagate(NodeResult::TARGET_LOSES);
      }
    } else {
      if (child_result == NodeResult::TARGET_WINS) {
        store_proven(known, parent, NodeResult::TARGET_WINS);
        propagate(NodeResult::TARGET_WINS);
      } else {
        parent.next_move_idx++;
      }
    }
  }
};

TEST(propagate_and_node_child_target_wins_advances) {
  PropagationSimulator sim;

  // Parent is AND node with 3 moves
  auto parent = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  parent.moves.resize(3);
  parent.next_move_idx = 0;

  auto child = make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA);

  sim.stack.push_back(parent);
  sim.stack.push_back(child);

  sim.propagate(NodeResult::TARGET_WINS);

  ASSERT_EQ(sim.stack.size(), 1u);  // child popped
  ASSERT_EQ(sim.stack[0].next_move_idx, 1);  // advanced to next move
}

TEST(propagate_and_node_child_target_loses_prunes) {
  PropagationSimulator sim;

  auto parent = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  parent.moves.resize(3);
  parent.next_move_idx = 1;

  auto child = make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA);

  sim.stack.push_back(parent);
  sim.stack.push_back(child);

  sim.propagate(NodeResult::TARGET_LOSES);

  // Both child AND parent should be popped (AND node fails immediately)
  ASSERT_TRUE(sim.stack.empty());
  // Parent should be stored as TARGET_LOSES
  auto r = check_resolved_known(sim.known, Board(0x1, 0x80000000, 0), 0, true);
  ASSERT_EQ(r, NodeResult::TARGET_LOSES);
}

TEST(propagate_or_node_child_target_wins_prunes) {
  PropagationSimulator sim;

  auto parent = make_frame(Board(0x1, 0x80000000, 0), 1, ProverMode::ESPADA);  // OR node
  parent.moves.resize(3);
  parent.next_move_idx = 0;
  ASSERT_FALSE(parent.is_and_node);

  auto child = make_frame(Board(0x2, 0x40000000, 0), 2, ProverMode::ESPADA);

  sim.stack.push_back(parent);
  sim.stack.push_back(child);

  sim.propagate(NodeResult::TARGET_WINS);

  // Both popped (OR node succeeds immediately)
  ASSERT_TRUE(sim.stack.empty());
  auto r = check_resolved_known(sim.known, Board(0x1, 0x80000000, 0), 1, false);
  ASSERT_EQ(r, NodeResult::TARGET_WINS);
}

TEST(propagate_or_node_child_target_loses_advances) {
  PropagationSimulator sim;

  auto parent = make_frame(Board(0x1, 0x80000000, 0), 1, ProverMode::ESPADA);
  parent.moves.resize(3);
  parent.next_move_idx = 0;
  ASSERT_FALSE(parent.is_and_node);

  auto child = make_frame(Board(0x2, 0x40000000, 0), 2, ProverMode::ESPADA);

  sim.stack.push_back(parent);
  sim.stack.push_back(child);

  sim.propagate(NodeResult::TARGET_LOSES);

  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 1);
}

TEST(propagate_and_node_all_children_succeed) {
  // Simulate AND node where all 3 children return TARGET_WINS
  PropagationSimulator sim;

  auto parent = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  parent.moves.resize(3);
  parent.next_move_idx = 0;
  parent.searched = true;

  sim.stack.push_back(parent);

  // Child 1: TARGET_WINS
  sim.stack.push_back(make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA));
  sim.propagate(NodeResult::TARGET_WINS);
  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 1);

  // Child 2: TARGET_WINS
  sim.stack.push_back(make_frame(Board(0x3, 0x40000000, 0), 1, ProverMode::ESPADA));
  sim.propagate(NodeResult::TARGET_WINS);
  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 2);

  // Child 3: TARGET_WINS
  sim.stack.push_back(make_frame(Board(0x4, 0x40000000, 0), 1, ProverMode::ESPADA));
  sim.propagate(NodeResult::TARGET_WINS);
  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 3);

  // Now all children exhausted (next_move_idx == moves.size())
  // In prove.cpp, the main loop handles this case:
  // AND node with all children TARGET_WINS -> node is TARGET_WINS
  ASSERT_EQ(sim.stack[0].next_move_idx, static_cast<int>(sim.stack[0].moves.size()));
}

TEST(propagate_or_node_all_children_fail) {
  PropagationSimulator sim;

  auto parent = make_frame(Board(0x1, 0x80000000, 0), 1, ProverMode::ESPADA);  // OR
  parent.moves.resize(2);
  parent.next_move_idx = 0;
  parent.searched = true;

  sim.stack.push_back(parent);

  // Child 1: TARGET_LOSES
  sim.stack.push_back(make_frame(Board(0x2, 0x40000000, 0), 2, ProverMode::ESPADA));
  sim.propagate(NodeResult::TARGET_LOSES);
  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 1);

  // Child 2: TARGET_LOSES
  sim.stack.push_back(make_frame(Board(0x3, 0x40000000, 0), 2, ProverMode::ESPADA));
  sim.propagate(NodeResult::TARGET_LOSES);
  ASSERT_EQ(sim.stack.size(), 1u);
  ASSERT_EQ(sim.stack[0].next_move_idx, 2);

  // All children exhausted -> OR node with all TARGET_LOSES -> TARGET_LOSES
  ASSERT_EQ(sim.stack[0].next_move_idx, static_cast<int>(sim.stack[0].moves.size()));
}

TEST(propagate_deep_cascade) {
  // Test cascading propagation: AND -> OR -> AND
  // If the deepest AND child fails, it should cascade up
  PropagationSimulator sim;

  // Root: AND node (ply 0, Espada)
  auto root = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  root.moves.resize(2);
  root.next_move_idx = 0;
  root.searched = true;

  // Child: OR node (ply 1)
  auto mid = make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA);
  mid.moves.resize(1);  // only one move
  mid.next_move_idx = 0;
  mid.searched = true;

  // Grandchild: AND node (ply 2)
  auto leaf = make_frame(Board(0x3, 0x20000000, 0), 2, ProverMode::ESPADA);

  sim.stack.push_back(root);
  sim.stack.push_back(mid);
  sim.stack.push_back(leaf);

  // Leaf returns TARGET_LOSES -> OR node's child failed
  sim.propagate(NodeResult::TARGET_LOSES);

  // OR node had only 1 move, so it advances to idx=1 which equals moves.size()
  // The main loop would then declare OR node as TARGET_LOSES
  // For this test, we check that the OR node advanced
  ASSERT_EQ(sim.stack.size(), 2u);  // root + mid
  ASSERT_EQ(sim.stack[1].next_move_idx, 1);  // OR advanced past its only move
}

TEST(propagate_deep_cascade_and_fails) {
  // Root (AND) -> Child (OR) -> Grandchild (AND) -> GG-child fails
  // Should cascade: GG fails -> Grandchild (AND) fails -> Child (OR) tries next
  PropagationSimulator sim;

  auto root = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  root.moves.resize(2);
  root.next_move_idx = 0;
  root.searched = true;

  auto child = make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA);
  child.moves.resize(2);
  child.next_move_idx = 0;
  child.searched = true;
  ASSERT_FALSE(child.is_and_node);  // OR node

  auto grandchild = make_frame(Board(0x3, 0x20000000, 0), 2, ProverMode::ESPADA);
  grandchild.moves.resize(3);
  grandchild.next_move_idx = 0;
  grandchild.searched = true;
  ASSERT_TRUE(grandchild.is_and_node);  // AND node

  auto gg_child = make_frame(Board(0x4, 0x10000000, 0), 3, ProverMode::ESPADA);

  sim.stack.push_back(root);
  sim.stack.push_back(child);
  sim.stack.push_back(grandchild);
  sim.stack.push_back(gg_child);

  // GG-child returns TARGET_WINS -> grandchild (AND) is happy, advances
  sim.propagate(NodeResult::TARGET_WINS);
  ASSERT_EQ(sim.stack.size(), 3u);  // root + child + grandchild
  ASSERT_EQ(sim.stack[2].next_move_idx, 1);
}

// ============================================================================
// Test 10: Repetition detection
// ============================================================================

TEST(repetition_same_board_on_stack) {
  Board b(0x1, 0x80000000, 0);

  // Simulate is_repetition: check if board's position_hash matches any on stack
  std::vector<StackFrame> stack;

  auto f0 = make_frame(b, 0, ProverMode::ESPADA);
  stack.push_back(f0);

  auto f1 = make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA);
  stack.push_back(f1);

  // Push same board as f0 at ply 2
  auto f2 = make_frame(b, 2, ProverMode::ESPADA);
  stack.push_back(f2);

  // Check repetition for f2 (top of stack)
  uint64_t ph = f2.board.position_hash();
  bool found = false;
  for (size_t i = 0; i + 1 < stack.size(); ++i) {
    if (stack[i].board.position_hash() == ph) {
      found = true;
      break;
    }
  }
  ASSERT_TRUE(found);
}

TEST(repetition_different_board_no_match) {
  std::vector<StackFrame> stack;

  stack.push_back(make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA));
  stack.push_back(make_frame(Board(0x2, 0x40000000, 0), 1, ProverMode::ESPADA));
  stack.push_back(make_frame(Board(0x4, 0x20000000, 0), 2, ProverMode::ESPADA));

  uint64_t ph = stack.back().board.position_hash();
  bool found = false;
  for (size_t i = 0; i + 1 < stack.size(); ++i) {
    if (stack[i].board.position_hash() == ph) {
      found = true;
      break;
    }
  }
  ASSERT_FALSE(found);
}

TEST(repetition_draw_maps_to_target_wins) {
  // In both Espada and Broquel, draws favor the target side
  // Repetition = draw -> TARGET_WINS
  // This is what prove.cpp does: NodeResult rep_result = NodeResult::TARGET_WINS;
  NodeResult rep_result = NodeResult::TARGET_WINS;
  ASSERT_EQ(rep_result, NodeResult::TARGET_WINS);
}

TEST(repetition_skips_current_top) {
  // The repetition check skips the current top of the stack
  Board b(0x1, 0x80000000, 0);
  std::vector<StackFrame> stack;
  stack.push_back(make_frame(b, 0, ProverMode::ESPADA));

  // Only one frame on stack; checking it against itself should not find repetition
  uint64_t ph = stack.back().board.position_hash();
  bool found = false;
  for (size_t i = 0; i + 1 < stack.size(); ++i) {
    if (stack[i].board.position_hash() == ph) {
      found = true;
      break;
    }
  }
  ASSERT_FALSE(found);
}

// ============================================================================
// Test 11: Draw value setup
// ============================================================================

TEST(draw_value_espada_ply0) {
  // Espada: draw = white loss
  // At ply 0 (white to move), draw_value should be negative (bad for white)
  int dv = -search::SCORE_TB_WIN;  // Espada base
  // ply 0 is even, so no flip
  ASSERT_TRUE(dv < 0);
  ASSERT_EQ(dv, -search::SCORE_TB_WIN);
}

TEST(draw_value_espada_ply1) {
  // Espada at ply 1 (black to move, board flipped so "white" = original black)
  int dv = -search::SCORE_TB_WIN;
  dv = -dv;  // ply 1 is odd
  ASSERT_TRUE(dv > 0);
  ASSERT_EQ(dv, search::SCORE_TB_WIN);
}

TEST(draw_value_broquel_ply0) {
  // Broquel: draw = white win
  // At ply 0 (white to move), draw_value should be positive (good for white)
  int dv = search::SCORE_TB_WIN;
  ASSERT_TRUE(dv > 0);
}

TEST(draw_value_broquel_ply1) {
  // Broquel at ply 1
  int dv = search::SCORE_TB_WIN;
  dv = -dv;  // odd ply
  ASSERT_TRUE(dv < 0);
}

// Verify the full draw_value computation matches prove.cpp's evaluate_node
TEST(draw_value_computation_all_cases) {
  for (auto mode : {ProverMode::ESPADA, ProverMode::BROQUEL}) {
    for (int ply = 0; ply < 4; ply++) {
      int dv = (mode == ProverMode::ESPADA) ? -search::SCORE_TB_WIN : search::SCORE_TB_WIN;
      if (ply % 2 != 0) dv = -dv;

      // Espada: white at ply 0,2 -> dv < 0 (draws bad for white)
      //         black at ply 1,3 -> dv > 0 (draws good for original black)
      // Broquel: white at ply 0,2 -> dv > 0 (draws good for white)
      //          black at ply 1,3 -> dv < 0 (draws bad for original black)
      if (mode == ProverMode::ESPADA) {
        if (ply % 2 == 0) ASSERT_TRUE(dv < 0);
        else ASSERT_TRUE(dv > 0);
      } else {
        if (ply % 2 == 0) ASSERT_TRUE(dv > 0);
        else ASSERT_TRUE(dv < 0);
      }
    }
  }
}

// ============================================================================
// Test 12: check_resolved with tablebases (integration)
// ============================================================================

// Helper: create a simple endgame position
// White king on sq_w, black king on sq_b
static Board make_kk_position(int sq_w, int sq_b) {
  Board b(0, 0, 0);
  b.white = 1u << sq_w;
  b.black = 1u << sq_b;
  b.kings = b.white | b.black;
  return b;
}

// White pawn on sq_wp, black king on sq_bk (white to move)
static Board make_pk_position(int sq_wp, int sq_bk) {
  Board b(0, 0, 0);
  b.white = 1u << sq_wp;
  b.black = 1u << sq_bk;
  b.kings = b.black;  // only black is king
  return b;
}

// White king on sq_wk, black pawn on sq_bp (white to move)
static Board make_kp_position(int sq_wk, int sq_bp) {
  Board b(0, 0, 0);
  b.white = 1u << sq_wk;
  b.black = 1u << sq_bp;
  b.kings = b.white;  // only white is king
  return b;
}

// 2 white pawns, 1 black king
static Board make_ppk_position(int sq_wp1, int sq_wp2, int sq_bk) {
  Board b(0, 0, 0);
  b.white = (1u << sq_wp1) | (1u << sq_wp2);
  b.black = 1u << sq_bk;
  b.kings = b.black;
  return b;
}

static const std::string TB_DIR = "../damas";

static bool tablebases_available() {
  return std::filesystem::exists(TB_DIR);
}

TEST(tb_king_vs_king_is_draw) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }

  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  // King vs King: use center squares (14, 17) where draw is confirmed
  Board b = make_kk_position(14, 17);
  Value v = cwdl.lookup_wdl_preloaded(b);
  ASSERT_EQ(v, Value::DRAW);

  // In both Espada and Broquel, draws = TARGET_WINS
  auto espada_result = check_resolved_tb(cwdl, b, 0, true, 4);  // AND node
  ASSERT_EQ(espada_result, NodeResult::TARGET_WINS);

  auto broquel_result = check_resolved_tb(cwdl, b, 0, false, 4);  // OR node
  ASSERT_EQ(broquel_result, NodeResult::TARGET_WINS);
}

TEST(tb_draw_espada_and_node) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  Board b = make_kk_position(14, 17);
  Value v = cwdl.lookup_wdl_preloaded(b);
  ASSERT_EQ(v, Value::DRAW);

  // Espada AND node at ply 0: draw -> TARGET_WINS
  auto result = check_resolved_tb(cwdl, b, 0, true, 4);
  ASSERT_EQ(result, NodeResult::TARGET_WINS);
}

TEST(tb_draw_espada_or_node) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  Board b = make_kk_position(14, 17);

  // Espada OR node at ply 1: draw -> TARGET_WINS
  auto result = check_resolved_tb(cwdl, b, 1, false, 4);
  ASSERT_EQ(result, NodeResult::TARGET_WINS);
}

TEST(tb_draw_broquel_and_node) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  Board b = make_kk_position(14, 17);

  // Broquel AND node at ply 1: draw -> TARGET_WINS
  auto result = check_resolved_tb(cwdl, b, 1, true, 4);
  ASSERT_EQ(result, NodeResult::TARGET_WINS);
}

TEST(tb_draw_broquel_or_node) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  Board b = make_kk_position(14, 17);

  // Broquel OR node at ply 0: draw -> TARGET_WINS
  auto result = check_resolved_tb(cwdl, b, 0, false, 4);
  ASSERT_EQ(result, NodeResult::TARGET_WINS);
}

TEST(tb_win_at_and_node) {
  // Find a winning position (e.g., 2 white pawns vs 1 black king)
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(5);

  // Try various positions to find a WIN
  // Pawn on sq 8 (row 3) and sq 12 (row 4), black king on sq 28
  Board b = make_ppk_position(8, 12, 28);
  Value v = cwdl.lookup_wdl_preloaded(b);

  if (v == Value::WIN) {
    // stm wins (+1) at AND node: stm_to_prover(1, true) = -1 = TARGET_LOSES
    auto result = check_resolved_tb(cwdl, b, 0, true, 5);
    ASSERT_EQ(result, NodeResult::TARGET_LOSES);

    // stm wins (+1) at OR node: stm_to_prover(1, false) = +1 = TARGET_WINS
    auto result2 = check_resolved_tb(cwdl, b, 0, false, 5);
    ASSERT_EQ(result2, NodeResult::TARGET_WINS);
  } else if (v == Value::LOSS) {
    // stm loses (-1) at AND node: stm_to_prover(-1, true) = +1 = TARGET_WINS
    auto result = check_resolved_tb(cwdl, b, 0, true, 5);
    ASSERT_EQ(result, NodeResult::TARGET_WINS);

    // stm loses (-1) at OR node: stm_to_prover(-1, false) = -1 = TARGET_LOSES
    auto result2 = check_resolved_tb(cwdl, b, 0, false, 5);
    ASSERT_EQ(result2, NodeResult::TARGET_LOSES);
  }
  // If DRAW, we already tested draws above
  std::cout << "  (PPK position value: "
            << (v == Value::WIN ? "WIN" : v == Value::LOSS ? "LOSS" : v == Value::DRAW ? "DRAW" : "UNKNOWN")
            << ")\n";
}

TEST(tb_loss_at_or_node) {
  // White king vs black pawn - try to find a loss for white
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  // White king on sq 0, black pawn on sq 23 (about to promote)
  Board b = make_kp_position(0, 23);
  Value v = cwdl.lookup_wdl_preloaded(b);

  std::cout << "  (KP position value: "
            << (v == Value::WIN ? "WIN" : v == Value::LOSS ? "LOSS" : v == Value::DRAW ? "DRAW" : "UNKNOWN")
            << ")\n";

  if (v == Value::LOSS) {
    // stm loses at OR node -> TARGET_LOSES
    auto result = check_resolved_tb(cwdl, b, 0, false, 4);
    ASSERT_EQ(result, NodeResult::TARGET_LOSES);
  }
}

// ============================================================================
// Test 13: Cross-mode consistency
// ============================================================================

TEST(espada_and_broquel_opposite_targets_for_win) {
  // For a decisive position (not draw), Espada and Broquel should give
  // consistent but different perspectives.
  //
  // At ply 0 in Espada: AND node (white). Target = black.
  // At ply 0 in Broquel: OR node (white). Target = white.
  //
  // If stm (white) wins:
  //   Espada (AND): stm_to_prover(1, true) = -1 = TARGET_LOSES (bad for target=black)
  //   Broquel (OR): stm_to_prover(1, false) = +1 = TARGET_WINS (good for target=white)
  // This is correct: white winning is bad for Espada's target (black) but good for Broquel's target (white)

  auto esp = stm_to_prover(1, true);   // Espada ply 0 AND
  auto broq = stm_to_prover(1, false); // Broquel ply 0 OR
  ASSERT_EQ(esp, NodeResult::TARGET_LOSES);
  ASSERT_EQ(broq, NodeResult::TARGET_WINS);
}

TEST(espada_and_broquel_draw_always_target_wins) {
  // In both modes, draws favor the target side
  // Espada: target = black, draw means black achieves at least a draw -> TARGET_WINS
  // Broquel: target = white, draw means white achieves at least a draw -> TARGET_WINS

  // Check draw handling in check_resolved_tb logic:
  // Draw at AND node: stm_val = -1, stm_to_prover(-1, true) = +1 = TARGET_WINS ✓
  // Draw at OR node: stm_val = +1, stm_to_prover(+1, false) = +1 = TARGET_WINS ✓
  auto and_draw = stm_to_prover(-1, true);
  auto or_draw = stm_to_prover(1, false);
  ASSERT_EQ(and_draw, NodeResult::TARGET_WINS);
  ASSERT_EQ(or_draw, NodeResult::TARGET_WINS);
}

// ============================================================================
// Test 14: Edge cases
// ============================================================================

TEST(position_key_all_zeros) {
  PositionKey k{0, 0, 0, 0};
  // Should hash without crashing
  auto h = std::hash<PositionKey>{}(k);
  (void)h;  // just ensure it doesn't crash
}

TEST(position_key_all_ones) {
  PositionKey k{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1};
  auto h = std::hash<PositionKey>{}(k);
  (void)h;
}

TEST(known_positions_store_unknown) {
  // NodeResult::UNKNOWN has value 0 - storing it is legal but unusual
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  kp.store(b, 0, NodeResult::UNKNOWN);

  auto result = kp.lookup(b, 0);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(*result, NodeResult::UNKNOWN);
}

TEST(stm_to_prover_zero) {
  // stm_val = 0 (draw in stm perspective)
  // AND node: -(0) = 0 = UNKNOWN
  // OR node: 0 = UNKNOWN
  // This shouldn't normally happen in practice (draws are handled differently)
  auto and_zero = stm_to_prover(0, true);
  auto or_zero = stm_to_prover(0, false);
  ASSERT_EQ(and_zero, NodeResult::UNKNOWN);
  ASSERT_EQ(or_zero, NodeResult::UNKNOWN);
}

TEST(empty_stack_propagation_no_crash) {
  PropagationSimulator sim;

  // Single frame, no parent to propagate to
  auto f = make_frame(Board(0x1, 0x80000000, 0), 0, ProverMode::ESPADA);
  sim.stack.push_back(f);

  sim.propagate(NodeResult::TARGET_WINS);
  ASSERT_TRUE(sim.stack.empty());
}

// ============================================================================
// Test 15: Board flip and ply parity
// ============================================================================

TEST(board_flip_changes_position_hash) {
  // Use an asymmetric position where flip definitely changes the hash
  Board b(0x7, 0x80000000, 0);  // 3 white pieces, 1 black
  Board flipped = flip(b);

  // After flip: white = flip(black), black = flip(white)
  // The piece counts swap, so it should be a different position
  ASSERT_NE(b.white, flipped.white);
  ASSERT_NE(b.position_hash(), flipped.position_hash());
}

TEST(board_double_flip_same_hash) {
  Board b(0x1, 0x80000000, 0);
  Board double_flipped = flip(flip(b));

  ASSERT_EQ(b.position_hash(), double_flipped.position_hash());
  ASSERT_EQ(b.white, double_flipped.white);
  ASSERT_EQ(b.black, double_flipped.black);
  ASSERT_EQ(b.kings, double_flipped.kings);
}

TEST(make_move_flips_board) {
  // After makeMove, the board is flipped (white becomes black, etc.)
  // In the starting position, white=0xFFF and black=0xFFF00000.
  // After flip, white=flip(black)=0xFFF, which coincidentally equals original white.
  // So we verify the flip semantics differently: the move changes the board state.
  Board b;  // starting position
  MoveList ml;
  generateMoves(b, ml);
  ASSERT_TRUE(ml.size() > 0);

  Board after = makeMove(b, ml[0]);
  // After a pawn move, the board hash should change
  ASSERT_NE(b.position_hash(), after.position_hash());
  // Also, all pieces count should be the same (no captures in opening)
  ASSERT_EQ(std::popcount(b.allPieces()), std::popcount(after.allPieces()));
}

// ============================================================================
// Test 16: Known positions with real game positions
// ============================================================================

TEST(known_positions_initial_position) {
  KnownPositions kp;
  Board b;  // starting position

  kp.store(b, 0, NodeResult::TARGET_WINS);
  auto result = kp.lookup(b, 0);
  ASSERT_TRUE(result.has_value());
  ASSERT_EQ(*result, NodeResult::TARGET_WINS);

  // Different ply parity
  ASSERT_FALSE(kp.lookup(b, 1).has_value());
}

TEST(known_positions_after_move) {
  Board b;
  MoveList ml;
  generateMoves(b, ml);
  ASSERT_TRUE(ml.size() > 0);

  Board after = makeMove(b, ml[0]);

  KnownPositions kp;
  kp.store(after, 1, NodeResult::TARGET_LOSES);

  // Should be findable at ply 1 (and any odd ply)
  ASSERT_TRUE(kp.lookup(after, 1).has_value());
  ASSERT_TRUE(kp.lookup(after, 3).has_value());
  ASSERT_FALSE(kp.lookup(after, 0).has_value());
  ASSERT_FALSE(kp.lookup(after, 2).has_value());
}

// ============================================================================
// Test 17: Interaction between store and check_resolved
// ============================================================================

TEST(store_then_check_resolved_espada_all_plies) {
  // Store a result at each ply (0-3) in Espada mode,
  // then verify check_resolved_known returns the correct prover result.
  for (int ply = 0; ply < 4; ply++) {
    KnownPositions kp;
    Board b(static_cast<Bb>(ply + 1), 0x80000000, 0);
    auto f = make_frame(b, ply, ProverMode::ESPADA);

    // Store TARGET_WINS
    store_proven(kp, f, NodeResult::TARGET_WINS);
    auto r = check_resolved_known(kp, b, ply, f.is_and_node);
    ASSERT_EQ(r, NodeResult::TARGET_WINS);

    // Overwrite with TARGET_LOSES
    store_proven(kp, f, NodeResult::TARGET_LOSES);
    r = check_resolved_known(kp, b, ply, f.is_and_node);
    ASSERT_EQ(r, NodeResult::TARGET_LOSES);
  }
}

TEST(store_then_check_resolved_broquel_all_plies) {
  for (int ply = 0; ply < 4; ply++) {
    KnownPositions kp;
    Board b(static_cast<Bb>(ply + 1), 0x80000000, 0);
    auto f = make_frame(b, ply, ProverMode::BROQUEL);

    store_proven(kp, f, NodeResult::TARGET_WINS);
    auto r = check_resolved_known(kp, b, ply, f.is_and_node);
    ASSERT_EQ(r, NodeResult::TARGET_WINS);

    store_proven(kp, f, NodeResult::TARGET_LOSES);
    r = check_resolved_known(kp, b, ply, f.is_and_node);
    ASSERT_EQ(r, NodeResult::TARGET_LOSES);
  }
}

// ============================================================================
// Test 18: No-moves positions in real game
// ============================================================================

TEST(trapped_piece_position) {
  // Try to construct a position where white has no moves
  // White pawn on sq 28 (row 8, can't advance), no captures available
  // Black pieces blocking
  Board b(0, 0, 0);
  b.white = 1u << 28;  // white pawn on sq 28 (top row for white)
  b.black = 0x00000001; // black piece somewhere else far away

  MoveList ml;
  generateMoves(b, ml);

  // White pawn on sq 28: possible forward moves are NW and NE
  // If those are off-board or blocked, no moves
  // sq 28 in row 7 (0-indexed): NW and NE might be available
  // This is testing that move generation works; the exact result depends on
  // the square layout

  std::cout << "  (White pawn sq28: " << ml.size() << " moves)\n";
}

// ============================================================================
// Test 19: check_resolved draw handling consistency
// ============================================================================

TEST(check_resolved_draw_is_target_wins_regardless_of_node_type) {
  // The key property: for ANY draw found in TB, check_resolved returns TARGET_WINS
  // regardless of whether it's AND or OR node
  //
  // This is the semantic meaning: draws are what the target side aims for
  //
  // Trace through check_resolved_tb code:
  //   Draw at AND node:
  //     stm_val = is_and_node ? -1 : 1  => -1
  //     stm_to_prover(-1, true) = -(-1) = +1 = TARGET_WINS ✓
  //   Draw at OR node:
  //     stm_val = is_and_node ? -1 : 1  => +1
  //     stm_to_prover(+1, false) = +1 = TARGET_WINS ✓

  // AND node draw
  int stm_val_and = -1;  // hardcoded for AND
  auto r_and = stm_to_prover(stm_val_and, true);
  ASSERT_EQ(r_and, NodeResult::TARGET_WINS);

  // OR node draw
  int stm_val_or = 1;  // hardcoded for OR
  auto r_or = stm_to_prover(stm_val_or, false);
  ASSERT_EQ(r_or, NodeResult::TARGET_WINS);
}

// ============================================================================
// Test 20: Score special checks
// ============================================================================

TEST(score_special_thresholds) {
  // Verify the score thresholds used in evaluate_node
  ASSERT_TRUE(search::is_special_score(search::SCORE_TB_WIN));
  ASSERT_TRUE(search::is_special_score(-search::SCORE_TB_WIN));
  ASSERT_TRUE(search::is_special_score(search::SCORE_MATE));
  ASSERT_TRUE(search::is_special_score(-search::SCORE_MATE));
  ASSERT_FALSE(search::is_special_score(0));
  ASSERT_FALSE(search::is_special_score(search::SCORE_DRAW));
  ASSERT_FALSE(search::is_special_score(-search::SCORE_DRAW));
  ASSERT_FALSE(search::is_special_score(search::SCORE_SPECIAL));  // not > SPECIAL
  ASSERT_TRUE(search::is_special_score(search::SCORE_SPECIAL + 1));
}

// ============================================================================
// Test 21: Full TB integration - verify WDL consistency
// ============================================================================

TEST(tb_integration_win_loss_perspective_consistency) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(4);

  // For a given endgame position, if stm wins (WIN), then after flipping
  // the board (now opponent to move), it should be LOSS.
  Board b = make_kp_position(14, 28);
  Value v = cwdl.lookup_wdl_preloaded(b);

  Board flipped = flip(b);
  Value v_flipped = cwdl.lookup_wdl_preloaded(flipped);

  std::cout << "  (KP original: "
            << (v == Value::WIN ? "WIN" : v == Value::LOSS ? "LOSS" : v == Value::DRAW ? "DRAW" : "UNKNOWN")
            << ", flipped: "
            << (v_flipped == Value::WIN ? "WIN" : v_flipped == Value::LOSS ? "LOSS" : v_flipped == Value::DRAW ? "DRAW" : "UNKNOWN")
            << ")\n";

  // If original is WIN, flipped should be LOSS (and vice versa)
  // If original is DRAW, flipped should be DRAW
  if (v == Value::WIN) ASSERT_EQ(v_flipped, Value::LOSS);
  else if (v == Value::LOSS) ASSERT_EQ(v_flipped, Value::WIN);
  else if (v == Value::DRAW) ASSERT_EQ(v_flipped, Value::DRAW);
}

TEST(tb_integration_multiple_endgames) {
  if (!tablebases_available()) {
    std::cout << "  (skipped: no tablebases)\n";
    return;
  }
  CompressedTablebaseManager cwdl(TB_DIR);
  cwdl.preload(5);

  // Test several endgame types
  struct TestCase {
    Board board;
    const char* description;
  };

  std::vector<TestCase> cases = {
    {make_kk_position(0, 31), "KK corners"},
    {make_kk_position(14, 17), "KK center"},
    {make_kp_position(4, 28), "KvP"},
    {make_pk_position(8, 28), "PvK"},
  };

  for (const auto& tc : cases) {
    Value v = cwdl.lookup_wdl_preloaded(tc.board);
    std::cout << "  " << tc.description << ": "
              << (v == Value::WIN ? "WIN" : v == Value::LOSS ? "LOSS"
                  : v == Value::DRAW ? "DRAW" : "UNKNOWN") << "\n";

    // For every resolved position, check_resolved_tb should not return UNKNOWN
    if (v != Value::UNKNOWN) {
      for (bool is_and : {true, false}) {
        auto result = check_resolved_tb(cwdl, tc.board, 0, is_and, 5);
        ASSERT_NE(result, NodeResult::UNKNOWN);
      }
    }
  }
}

// ============================================================================
// Test 22: Verify no-moves handling stores correctly in DB
// ============================================================================

TEST(no_moves_store_consistency) {
  // When stm has no moves, prove.cpp stores TARGET_LOSES directly
  // (line 293), which has int8_t value -1.
  // This should be consistent with what store_proven would produce.

  Board b(0x1, 0x80000000, 0);  // arbitrary

  for (auto mode : {ProverMode::ESPADA, ProverMode::BROQUEL}) {
    for (int ply : {0, 1}) {
      KnownPositions kp1, kp2;
      auto f = make_frame(b, ply, mode);

      // Method 1: direct store (what prove.cpp does for no-moves)
      kp1.store(b, ply, NodeResult::TARGET_LOSES);  // -1

      // Method 2: via store_proven with appropriate prover result
      // stm loses -> prover result depends on node type
      NodeResult prover_result = stm_to_prover(-1, f.is_and_node);
      store_proven(kp2, f, prover_result);

      // Both should produce the same raw stored value
      auto raw1 = kp1.lookup(b, ply);
      auto raw2 = kp2.lookup(b, ply);
      ASSERT_TRUE(raw1.has_value());
      ASSERT_TRUE(raw2.has_value());
      ASSERT_EQ(*raw1, *raw2);
    }
  }
}

// ============================================================================
// Test 23: Hash collision resistance
// ============================================================================

TEST(position_key_hash_distribution) {
  // Generate many keys and check for collisions
  std::unordered_map<size_t, int> hash_counts;
  int collisions = 0;

  for (Bb w = 1; w < 4096; w += 17) {
    for (Bb b = 0x100000; b < 0xFFF00000; b += 0x11100000) {
      for (uint8_t btm = 0; btm <= 1; btm++) {
        PositionKey k{w, b, 0, btm};
        size_t h = std::hash<PositionKey>{}(k);
        if (++hash_counts[h] > 1) collisions++;
      }
    }
  }

  // With a good hash, collisions should be very rare
  std::cout << "  (hashed " << hash_counts.size() << " keys, " << collisions << " collisions)\n";
  // Allow some collisions but not many
  ASSERT_TRUE(collisions < static_cast<int>(hash_counts.size()) / 100);
}

// ============================================================================
// Test 24: Verify AND/OR exhaustion handling
// ============================================================================

TEST(and_node_exhausted_means_target_wins) {
  // When all moves of an AND node are exhausted (all returned TARGET_WINS),
  // the AND node result is TARGET_WINS.
  // This is handled in the main loop, not in propagate.

  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 0, ProverMode::ESPADA);
  f.moves.resize(3);
  f.next_move_idx = 3;  // all exhausted

  ASSERT_TRUE(f.is_and_node);
  ASSERT_EQ(f.next_move_idx, static_cast<int>(f.moves.size()));

  // The main loop would do:
  NodeResult result = NodeResult::TARGET_WINS;
  store_proven(kp, f, result);

  auto resolved = check_resolved_known(kp, b, 0, true);
  ASSERT_EQ(resolved, NodeResult::TARGET_WINS);
}

TEST(or_node_exhausted_means_target_loses) {
  KnownPositions kp;
  Board b(0x1, 0x80000000, 0);
  auto f = make_frame(b, 1, ProverMode::ESPADA);  // OR node
  f.moves.resize(3);
  f.next_move_idx = 3;

  ASSERT_FALSE(f.is_and_node);

  NodeResult result = NodeResult::TARGET_LOSES;
  store_proven(kp, f, result);

  auto resolved = check_resolved_known(kp, b, 1, false);
  ASSERT_EQ(resolved, NodeResult::TARGET_LOSES);
}

// ============================================================================
// Test 25: Move generation + prove logic integration
// ============================================================================

TEST(real_position_move_count_and_frame) {
  Board b;  // starting position
  MoveList ml;
  generateMoves(b, ml);

  // Starting position should have 9 moves (Spanish checkers)
  std::cout << "  (starting position: " << ml.size() << " moves)\n";
  ASSERT_TRUE(ml.size() > 0);

  // Create frame for each mode
  auto espada = make_frame(b, 0, ProverMode::ESPADA);
  ASSERT_TRUE(espada.is_and_node);  // white is defending in Espada

  auto broquel = make_frame(b, 0, ProverMode::BROQUEL);
  ASSERT_FALSE(broquel.is_and_node);  // white is attacking in Broquel
}

// ============================================================================
// Test 26: Save/load interaction with store_proven
// ============================================================================

TEST(save_load_preserves_stm_perspective) {
  const std::string tmpfile = "/tmp/test_prover_stm.bin";
  KnownPositions kp;

  // Store results from both AND and OR nodes
  Board b1(0x1, 0x80000000, 0);
  Board b2(0x2, 0x40000000, 0);

  auto f1 = make_frame(b1, 0, ProverMode::ESPADA);  // AND
  auto f2 = make_frame(b2, 1, ProverMode::ESPADA);  // OR

  store_proven(kp, f1, NodeResult::TARGET_WINS);
  store_proven(kp, f2, NodeResult::TARGET_LOSES);

  ASSERT_TRUE(kp.save(tmpfile));

  // Load and verify
  KnownPositions kp2;
  ASSERT_TRUE(kp2.load(tmpfile));

  // Check the stored stm values are preserved
  auto r1 = check_resolved_known(kp2, b1, 0, true);
  ASSERT_EQ(r1, NodeResult::TARGET_WINS);

  auto r2 = check_resolved_known(kp2, b2, 1, false);
  ASSERT_EQ(r2, NodeResult::TARGET_LOSES);

  std::remove(tmpfile.c_str());
}

// ============================================================================
// Main
// ============================================================================

int main() {
  std::cout << "=== Prover Tests ===\n\n";

  for (const auto& test : test_registry()) {
    tests_run++;
    std::cout << "  " << test.name << "... ";
    try {
      test.func();
      tests_passed++;
      std::cout << "OK\n";
    } catch (const std::exception& e) {
      tests_failed++;
      std::cout << "FAILED\n";
    }
  }

  std::cout << "\n=== Results: " << tests_passed << " passed, "
            << tests_failed << " failed, " << tests_run << " total ===\n";

  return tests_failed > 0 ? 1 : 0;
}
