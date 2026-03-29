// Game prover using Espada/Broquel decomposition
//
// Proves the theoretical value of Spanish Checkers positions by decomposing
// the 3-valued game (W/D/L) into two 2-valued games:
//   Espada: draw = white loss (proves black can force at least a draw)
//   Broquel: draw = white win (proves white can force at least a draw)
//
// Uses AND/OR tree search with Torquemada engine for move selection and
// compressed WDL tablebases for leaf evaluation.

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "search/search.hpp"
#include "tablebase/compression.hpp"
#include <atomic>
#include <bit>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Types
// ============================================================================

enum class ProverMode { ESPADA, BROQUEL };

// Result from the prover's perspective:
//   TARGET_WINS = the target side wins (black in Espada, white in Broquel)
//   TARGET_LOSES = the target side loses
enum class NodeResult : int8_t { TARGET_WINS = 1, TARGET_LOSES = -1, UNKNOWN = 0 };

// ============================================================================
// Known positions database
// ============================================================================

// Key for proven positions: actual board data + side to move.
// Only positions with n_reversible == 0 are stored, to avoid history dependence.
struct PositionKey {
  Bb white, black, kings;
  uint8_t black_to_move;  // 1 if original black is to move, 0 otherwise

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

private:
  std::unordered_map<PositionKey, NodeResult> map_;
  size_t new_count_ = 0;
};

// ============================================================================
// Stack frame for AND/OR tree search
// ============================================================================

struct StackFrame {
  Board board;
  int game_ply;         // 0 = root; even = original white, odd = original black
  bool is_and_node;     // true = must prove ALL children, false = need ONE child
  std::vector<Move> moves;
  std::vector<int> move_scores;  // search scores for move ordering
  int next_move_idx = 0;
  bool searched = false;
};

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> g_stop_requested{false};

static void sigint_handler(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

// ============================================================================
// Move notation helper
// ============================================================================

static std::string move_string(const Board& board, const Move& move, bool white_to_move) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    if (fm.move == move && !fm.path.empty()) {
      std::string result;
      for (size_t i = 0; i < fm.path.size(); ++i) {
        if (i > 0) result += (move.isCapture() ? "x" : "-");
        int sq = fm.path[i];
        int display = white_to_move ? (sq + 1) : (32 - sq);
        result += std::to_string(display);
      }
      return result;
    }
  }
  return "???";
}

// ============================================================================
// Prover
// ============================================================================

struct ProverConfig {
  ProverMode mode = ProverMode::ESPADA;
  std::string tb_dir = "../damas";
  int tb_pieces = 8;
  std::string nn_model;      // NN model for OR nodes (strong eval)
  std::string and_model;     // Separate model for AND nodes (fast eval)
  std::string known_file;
  std::string opening;       // optional opening moves (e.g. "1. 10-14 22-18")
  double or_time = 3.0;   // seconds for OR node search
  double and_time = 0.5;  // seconds for AND node search
  int tt_size = 256;      // MB
  bool verbose = false;
  Board start_board;       // initial position
  int start_ply = 0;       // game ply after applying opening moves
};

class Prover {
public:
  Prover(const ProverConfig& config)
      : config_(config), cwdl_(config.tb_dir),
        and_searcher_("", 0, config.and_model),  // AND nodes: fast linear model
        or_searcher_("", 0, config.nn_model) {   // OR nodes: full NN model
    // Set up AND searcher (fast eval for proving all moves lose)
    and_searcher_.set_tt_size(config.tt_size / 2);
    and_searcher_.set_stop_flag(&g_stop_requested);
    and_searcher_.set_verbose(false);

    // Set up OR searcher (NN eval for finding strong moves)
    or_searcher_.set_tt_size(config.tt_size / 2);
    or_searcher_.set_stop_flag(&g_stop_requested);
    or_searcher_.set_verbose(false);

    // Preload compressed tablebases
    std::cout << "Preloading compressed tablebases from " << config.tb_dir
              << " (up to " << config.tb_pieces << " pieces)...\n";
    cwdl_.preload(config.tb_pieces);

    // WDL probe for compressed tablebases only.
    // Known positions are NOT included here — they depend on game_ply parity
    // which the search doesn't track. Known positions are checked at the
    // prover level in check_resolved() instead.
    auto wdl_probe = [this](const Board& b) -> std::optional<int> {
      int pieces = std::popcount(b.allPieces());
      if (pieces <= config_.tb_pieces) {
        Value v = cwdl_.lookup_wdl_preloaded(b);
        switch (v) {
          case Value::WIN: return 1;
          case Value::LOSS: return -1;
          case Value::DRAW: return 0;  // draw_value_ handles remapping
          default: return std::nullopt;
        }
      }
      return std::nullopt;
    };
    and_searcher_.set_wdl_probe(wdl_probe, config.tb_pieces);
    or_searcher_.set_wdl_probe(wdl_probe, config.tb_pieces);

    // Load known positions if file exists
    if (!config.known_file.empty()) {
      if (known_.load(config.known_file)) {
        std::cout << "Loaded " << known_.size() << " known positions from "
                  << config.known_file << "\n";
      }
    }
  }

  void run() {
    auto start_time = std::chrono::steady_clock::now();

    // Initialize stack with starting position
    stack_.clear();
    stack_.push_back(make_frame(config_.start_board, config_.start_ply));

    std::cout << "\n=== Prover: " << (config_.mode == ProverMode::ESPADA ? "Espada" : "Broquel")
              << " mode ===\n";
    std::cout << "Proving from initial position...\n\n";

    while (!stack_.empty() && !g_stop_requested.load()) {
      auto& frame = stack_.back();

      // 1. Check if position is already resolved
      NodeResult resolved = check_resolved(frame.board, frame.game_ply, frame.is_and_node);
      if (resolved != NodeResult::UNKNOWN) {
        propagate(resolved);
        continue;
      }

      // 2. Check game-level repetition
      if (is_repetition(frame.board)) {
        // Draw by repetition -> map to mode result, do NOT store
        NodeResult rep_result = NodeResult::TARGET_WINS;  // draws favor target side
        if (config_.verbose) {
          std::cout << std::string(frame.game_ply * 2, ' ')
                    << "Repetition detected at ply " << frame.game_ply << " -> "
                    << (rep_result == NodeResult::TARGET_WINS ? "target wins" : "target loses")
                    << "\n";
        }
        propagate(rep_result);
        continue;
      }

      // 3. Generate and evaluate moves (first visit)
      if (!frame.searched) {
        MoveList ml;
        generateMoves(frame.board, ml);
        if (ml.empty()) {
          // No legal moves = side to move loses
          // stm loses -> stm_val = -1 -> store in DB
          known_.store(frame.board, frame.game_ply, NodeResult::TARGET_LOSES);  // -1 = stm loses
          NodeResult prover_result = stm_to_prover(-1, frame.is_and_node);
          if (config_.verbose) {
            std::cout << std::string(frame.game_ply * 2, ' ')
                      << "No moves at ply " << frame.game_ply << " -> stm loses\n";
          }
          propagate(prover_result);
          continue;
        }

        // Run search to order moves and possibly resolve immediately
        if (evaluate_node(frame, ml)) {
          continue;  // Node was resolved by search
        }
      }

      // 4. Expand next child
      if (frame.next_move_idx >= static_cast<int>(frame.moves.size())) {
        // All moves exhausted
        if (frame.is_and_node) {
          // AND node: all children had target result -> this node is target wins
          NodeResult result = NodeResult::TARGET_WINS;
          store_proven(frame, result);
          print_proven(frame, result);
          propagate(result);
        } else {
          // OR node: no child had target result -> this node is target loses
          NodeResult result = NodeResult::TARGET_LOSES;
          store_proven(frame, result);
          print_proven(frame, result);
          propagate(result);
        }
        continue;
      }

      // Push child
      Move move = frame.moves[frame.next_move_idx];
      Board child_board = makeMove(frame.board, move);

      if (config_.verbose) {
        bool white_to_move = (frame.game_ply % 2 == 0);
        std::cout << std::string(frame.game_ply * 2, ' ');
        if (white_to_move) {
          std::cout << (frame.game_ply / 2 + 1) << ". ";
        } else {
          std::cout << (frame.game_ply / 2 + 1) << "... ";
        }
        std::cout << move_string(frame.board, move, white_to_move);
        std::cout << " [" << (frame.is_and_node ? "AND" : "OR")
                  << " " << (frame.next_move_idx + 1) << "/" << frame.moves.size() << "]\n";
      }

      stack_.push_back(make_frame(child_board, frame.game_ply + 1));
    }

    // Save on exit
    save_known();

    auto elapsed = std::chrono::steady_clock::now() - start_time;
    double secs = std::chrono::duration<double>(elapsed).count();

    std::cout << "\n=== Prover finished ===\n";
    std::cout << "Known positions: " << known_.size() << "\n";
    std::cout << "Time: " << secs << "s\n";

    if (stack_.empty()) {
      auto stm_result = known_.lookup(config_.start_board, config_.start_ply);
      if (stm_result) {
        // Root is at ply 0, AND node in Espada (OR in Broquel).
        // DB stores stm perspective. Convert to prover perspective.
        bool root_is_and = (config_.mode == ProverMode::ESPADA);
        NodeResult prover_result = stm_to_prover(static_cast<int>(*stm_result), root_is_and);
        std::cout << "ROOT PROVEN: ";
        if (prover_result == NodeResult::TARGET_WINS) {
          if (config_.mode == ProverMode::ESPADA) {
            std::cout << "Black wins Espada (black can force at least a draw)\n";
          } else {
            std::cout << "White wins Broquel (white can force at least a draw)\n";
          }
        } else {
          if (config_.mode == ProverMode::ESPADA) {
            std::cout << "White wins Espada (white can force a win)\n";
          } else {
            std::cout << "Black wins Broquel (black can force a win)\n";
          }
        }
      }
    } else {
      std::cout << "Search interrupted at stack depth " << stack_.size() << "\n";
    }
  }

private:
  StackFrame make_frame(const Board& board, int game_ply) {
    StackFrame f;
    f.board = board;
    f.game_ply = game_ply;
    // In Espada: even ply = original white = AND node (must prove all moves lose)
    //            odd ply = original black = OR node (need one winning move)
    // In Broquel: reversed
    bool even_ply = (game_ply % 2 == 0);
    f.is_and_node = (config_.mode == ProverMode::ESPADA) ? even_ply : !even_ply;
    return f;
  }

  // Convert stm-perspective value (+1/-1) to prover-perspective (TARGET_WINS/LOSES).
  // Known positions DB and WDL probes use stm perspective.
  // AND/OR tree propagation uses prover perspective.
  //
  // AND node (defending side to move): stm losing = prover wins -> negate
  // OR node (attacking side to move): stm winning = prover wins -> keep
  static NodeResult stm_to_prover(int stm_val, bool is_and_node) {
    int prover_val = is_and_node ? -stm_val : stm_val;
    return static_cast<NodeResult>(prover_val);
  }

  // Convert prover-perspective result to stm-perspective for DB storage.
  // Inverse of stm_to_prover.
  static int prover_to_stm(NodeResult result, bool is_and_node) {
    int prover_val = static_cast<int>(result);
    return is_and_node ? -prover_val : prover_val;
  }

  // Store a proven result (prover perspective) in the known DB (stm perspective).
  void store_proven(const StackFrame& frame, NodeResult result) {
    int stm_val = prover_to_stm(result, frame.is_and_node);
    known_.store(frame.board, frame.game_ply, static_cast<NodeResult>(stm_val));
  }

  // Check if position is already resolved. Returns prover-perspective result.
  NodeResult check_resolved(const Board& board, int game_ply, bool is_and_node) {
    // Check known positions (stored in stm perspective, only for n_reversible == 0)
    if (auto r = known_.lookup(board, game_ply)) {
      return stm_to_prover(static_cast<int>(*r), is_and_node);
    }

    // Check tablebases
    int pieces = std::popcount(board.allPieces());
    if (pieces <= config_.tb_pieces) {
      Value v = cwdl_.lookup_wdl_preloaded(board);
      int stm_val;
      switch (v) {
        case Value::WIN: stm_val = 1; break;
        case Value::LOSS: stm_val = -1; break;
        case Value::DRAW:
          // Draw = target wins. Convert to stm perspective first:
          // AND node: target wins -> stm loses -> stm_val = -1
          // OR node: target wins -> stm wins -> stm_val = +1
          stm_val = is_and_node ? -1 : 1;
          break;
        default: return NodeResult::UNKNOWN;
      }
      return stm_to_prover(stm_val, is_and_node);
    }
    return NodeResult::UNKNOWN;
  }

  bool is_repetition(const Board& board) {
    uint64_t ph = board.position_hash();
    // Check all positions on the stack (skip current top)
    for (size_t i = 0; i + 1 < stack_.size(); ++i) {
      if (stack_[i].board.position_hash() == ph) return true;
    }
    return false;
  }

  // Evaluate a node with Torquemada search. Returns true if the node was resolved.
  bool evaluate_node(StackFrame& frame, const MoveList& /*ml*/) {
    frame.searched = true;

    // Select searcher: AND nodes use fast PST, OR nodes use NN
    auto& searcher = frame.is_and_node ? and_searcher_ : or_searcher_;

    // Set draw_value based on mode and game ply
    int dv = (config_.mode == ProverMode::ESPADA) ? -search::SCORE_TB_WIN : search::SCORE_TB_WIN;
    if (frame.game_ply % 2 != 0) dv = -dv;
    searcher.set_draw_value(dv);

    double time = frame.is_and_node ? config_.and_time : config_.or_time;
    auto tc = search::TimeControl::with_time(time, time * 3);

    g_stop_requested.store(false);
    auto result = searcher.search_multi(frame.board, 100, tc, 200);
    g_stop_requested.store(false);

    // Copy moves in search-score order
    frame.moves.clear();
    frame.move_scores.clear();
    for (const auto& ms : result.moves) {
      frame.moves.push_back(ms.move);
      frame.move_scores.push_back(ms.score);
    }

    // Check if the best move already has a decisive score
    if (!result.moves.empty()) {
      int best_score = result.moves[0].score;

      // For AND nodes: if white's best move is a win (> SCORE_SPECIAL), then
      // the AND node might fail (target loses at this node).
      // But we can't conclude yet — we need to check if ALL moves are proven.
      // Only if the WORST move is still decisive can we conclude.
      //
      // For OR nodes: if the best move is decisive in target's favor, we're done.

      if (frame.is_and_node) {
        // AND node: check if ALL moves are proven as target wins (stm loses)
        // The worst move (last in sorted order) determines this.
        int worst_score = result.moves.back().score;
        if (search::is_special_score(worst_score) && worst_score < 0) {
          // All moves are proven losses for stm -> AND node proven, target wins
          NodeResult nr = NodeResult::TARGET_WINS;
          store_proven(frame, nr);
          print_proven(frame, nr);
          propagate(nr);
          return true;
        }
        if (search::is_special_score(best_score) && best_score > 0) {
          // Best move (from stm perspective) is a win -> AND node fails, target loses
          NodeResult nr = NodeResult::TARGET_LOSES;
          store_proven(frame, nr);
          print_proven(frame, nr);
          propagate(nr);
          return true;
        }
      } else {
        // OR node: check if best move is a win for stm (target wins at OR node)
        if (search::is_special_score(best_score) && best_score > 0) {
          // stm wins -> target wins at OR node
          NodeResult nr = NodeResult::TARGET_WINS;
          store_proven(frame, nr);
          print_proven(frame, nr);
          propagate(nr);
          return true;
        }
        // Check if ALL moves are proven losses for stm
        int worst_score = result.moves.back().score;
        if (search::is_special_score(worst_score) && worst_score < 0 &&
            search::is_special_score(best_score) && best_score < 0) {
          // All moves lose -> target loses at OR node
          NodeResult nr = NodeResult::TARGET_LOSES;
          store_proven(frame, nr);
          print_proven(frame, nr);
          propagate(nr);
          return true;
        }
      }
    }

    return false;  // Not resolved, need to expand children
  }

  void propagate(NodeResult child_result) {
    stack_.pop_back();
    if (stack_.empty()) return;

    auto& parent = stack_.back();

    if (parent.is_and_node) {
      // AND node: need all children to be TARGET_WINS
      if (child_result == NodeResult::TARGET_WINS) {
        parent.next_move_idx++;
        // Continue to next move (main loop will handle)
      } else {
        // One child is TARGET_LOSES -> AND node fails
        store_proven(parent, NodeResult::TARGET_LOSES);
        print_proven(parent, NodeResult::TARGET_LOSES);
        propagate(NodeResult::TARGET_LOSES);
      }
    } else {
      // OR node: need one child to be TARGET_WINS
      if (child_result == NodeResult::TARGET_WINS) {
        store_proven(parent, NodeResult::TARGET_WINS);
        print_proven(parent, NodeResult::TARGET_WINS);
        propagate(NodeResult::TARGET_WINS);
      } else {
        parent.next_move_idx++;
        // Continue to next move
      }
    }

    // Auto-save periodically
    if (known_.new_count() >= 100) {
      save_known();
    }
  }

  // Build the current game line from the stack as a string like:
  // "1. 10-14 22-18 2. 5-10 23-20 3. ..."
  std::string current_line() const {
    std::string line;
    for (size_t i = 0; i + 1 < stack_.size(); ++i) {
      const auto& f = stack_[i];
      if (f.next_move_idx >= static_cast<int>(f.moves.size())) continue;
      Move move = f.moves[f.next_move_idx];
      bool wtm = (f.game_ply % 2 == 0);
      if (wtm) {
        if (!line.empty()) line += ' ';
        line += std::to_string(f.game_ply / 2 + 1) + ". ";
      } else {
        if (line.empty()) {
          line += std::to_string(f.game_ply / 2 + 1) + "... ";
        } else {
          line += ' ';
        }
      }
      line += move_string(f.board, move, wtm);
    }
    return line;
  }

  void print_proven(const StackFrame& frame, NodeResult result) {
    bool white_to_move = (frame.game_ply % 2 == 0);
    std::cout << "[ply " << frame.game_ply
              << " | depth " << stack_.size()
              << " | known " << known_.size() << "] "
              << (white_to_move ? "W" : "B") << " to move: "
              << (frame.is_and_node ? "AND" : "OR") << " node -> "
              << (result == NodeResult::TARGET_WINS ? "TARGET_WINS" : "TARGET_LOSES")
              << "\n  " << current_line() << "\n";
  }

  void save_known() {
    if (config_.known_file.empty()) return;
    if (known_.save(config_.known_file)) {
      std::cout << "[Saved " << known_.size() << " known positions to "
                << config_.known_file << "]\n";
      known_.reset_new_count();
    }
  }

  ProverConfig config_;
  CompressedTablebaseManager cwdl_;
  search::Searcher and_searcher_;  // Fast model for AND nodes
  search::Searcher or_searcher_;   // Full NN model for OR nodes
  KnownPositions known_;
  std::vector<StackFrame> stack_;
};

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "\nSpanish Checkers game prover using Espada/Broquel decomposition.\n"
            << "\nOptions:\n"
            << "  --mode espada|broquel   Proof mode (default: espada)\n"
            << "  --tb-dir PATH           CWDL tablebase directory (default: ../damas)\n"
            << "  --tb-pieces N           Max pieces for WDL TB (default: 8)\n"
            << "  --model PATH            Neural network model for OR nodes (default: none)\n"
            << "  --and-model PATH        Fast model for AND nodes (default: same as --model)\n"
            << "  --known-file PATH       Known positions database file\n"
            << "  --or-time SECONDS       Time for OR node search (default: 3.0)\n"
            << "  --and-time SECONDS      Time for AND node search (default: 0.5)\n"
            << "  --tt-size MB            Transposition table size (default: 256)\n"
            << "  --opening MOVES         Opening moves (e.g. \"1. 10-14 22-18\")\n"
            << "  --verbose               Print detailed game progress\n"
            << "  -h, --help              Show this help\n";
}

int main(int argc, char** argv) {
  ProverConfig config;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string mode = argv[++i];
      if (mode == "espada") config.mode = ProverMode::ESPADA;
      else if (mode == "broquel") config.mode = ProverMode::BROQUEL;
      else {
        std::cerr << "Unknown mode: " << mode << "\n";
        return 1;
      }
    } else if (arg == "--tb-dir" && i + 1 < argc) {
      config.tb_dir = argv[++i];
    } else if (arg == "--tb-pieces" && i + 1 < argc) {
      config.tb_pieces = std::stoi(argv[++i]);
    } else if (arg == "--model" && i + 1 < argc) {
      config.nn_model = argv[++i];
    } else if (arg == "--and-model" && i + 1 < argc) {
      config.and_model = argv[++i];
    } else if (arg == "--known-file" && i + 1 < argc) {
      config.known_file = argv[++i];
    } else if (arg == "--or-time" && i + 1 < argc) {
      config.or_time = std::stod(argv[++i]);
    } else if (arg == "--and-time" && i + 1 < argc) {
      config.and_time = std::stod(argv[++i]);
    } else if (arg == "--tt-size" && i + 1 < argc) {
      config.tt_size = std::stoi(argv[++i]);
    } else if (arg == "--opening" && i + 1 < argc) {
      config.opening = argv[++i];
    } else if (arg == "--verbose") {
      config.verbose = true;
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return 1;
    }
  }

  // Default AND model: same as OR model if not specified
  if (config.and_model.empty()) {
    config.and_model = config.nn_model;
  }

  // Default known file based on mode
  if (config.known_file.empty()) {
    config.known_file = (config.mode == ProverMode::ESPADA)
                            ? "known_espada.bin"
                            : "known_broquel.bin";
  }

  // Apply opening moves if specified
  if (!config.opening.empty()) {
    GameRecord record = parseGame(Board(), config.opening);
    if (!record.complete) {
      std::cerr << "Error parsing opening: " << record.error << "\n";
      return 1;
    }
    config.start_board = record.finalBoard;
    config.start_ply = static_cast<int>(record.moves.size());
    std::cout << "Opening (" << config.start_ply << " plies): " << config.opening << "\n";
  }

  std::signal(SIGINT, sigint_handler);

  Prover prover(config);
  prover.run();

  return 0;
}
