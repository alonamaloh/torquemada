// Interactive play using proven positions from the Espada/Broquel prover.
//
// The engine plays in a derived game (Espada or Broquel) using:
//   - Proven positions database (from the prover)
//   - Compressed WDL tablebases (with draw remapping)
//   - Neural network evaluation
//
// In Espada mode, draws count as losses for white, so the engine
// playing black should never lose (if it has enough proven positions).
// In Broquel mode, draws count as wins for white.
//
// Commands:
//   <move>  - Enter a move (e.g., "11-15" or "11x18x25")
//   b       - Take back the last move
//   2       - Toggle 2-player mode
//   m       - Assign engine to current side
//   n N     - Set search node limit
//   t S     - Set search time limit in seconds
//   q       - Quit
//   ?       - Show legal moves

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "search/search.hpp"
#include "tablebase/compression.hpp"
#include <atomic>
#include <bit>
#include <csignal>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Known positions database (same format as prove.cpp)
// ============================================================================

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
  std::optional<int> lookup(const Board& board, int game_ply) const {
    if (board.n_reversible != 0) return std::nullopt;
    PositionKey key{board.white, board.black, board.kings,
                    static_cast<uint8_t>(game_ply % 2)};
    auto it = map_.find(key);
    if (it != map_.end()) return static_cast<int>(it->second);
    return std::nullopt;
  }

  size_t size() const { return map_.size(); }

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
      map_[key] = v;
    }
    return in.good();
  }

private:
  std::unordered_map<PositionKey, int8_t> map_;
};

// ============================================================================
// Globals
// ============================================================================

static std::atomic<bool> g_stop_requested{false};

static void sigint_handler(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

// ============================================================================
// Game state
// ============================================================================

enum class Mode { ESPADA, BROQUEL };

struct GameState {
  Board board;
  std::vector<Board> history;
  std::unordered_map<uint64_t, int> position_counts;
  bool engine_plays_white = false;
  bool engine_plays_black = true;
  bool two_player_mode = false;
  int ply = 0;  // game ply (0 = original white to move)
  uint64_t nodes = 0;  // 0 = use time control
  double time_limit = 5.0;

  GameState() {
    position_counts[board.position_hash()] = 1;
  }
};

// ============================================================================
// Display helpers
// ============================================================================

static int square_to_display(int sq, bool white_to_move) {
  return white_to_move ? (sq + 1) : (32 - sq);
}

static int display_to_square(int display, bool white_to_move) {
  int sq = white_to_move ? display : (33 - display);
  return sq - 1;
}

static void print_board(const Board& board, bool white_to_move) {
  Board display = white_to_move ? board : flip(board);
  std::cout << "\n" << display;
  std::cout << (white_to_move ? "White" : "Black") << " to move\n";
}

static std::string move_to_string(const Board& board, const Move& move, bool white_to_move) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    if (fm.move == move && !fm.path.empty()) {
      std::string result = std::to_string(square_to_display(fm.path[0], white_to_move));
      for (size_t i = 1; i < fm.path.size(); ++i) {
        result += (move.isCapture() ? "x" : "-");
        result += std::to_string(square_to_display(fm.path[i], white_to_move));
      }
      return result;
    }
  }
  return "???";
}

static void show_legal_moves(const Board& board, bool white_to_move) {
  std::vector<FullMove> moves;
  generateFullMoves(board, moves);
  if (moves.empty()) {
    std::cout << "No legal moves - game over!\n";
    return;
  }
  std::cout << "Legal moves: ";
  for (size_t i = 0; i < moves.size(); ++i) {
    if (i > 0) std::cout << ", ";
    const auto& fm = moves[i];
    if (!fm.path.empty()) {
      std::cout << square_to_display(fm.path[0], white_to_move);
      for (size_t j = 1; j < fm.path.size(); ++j) {
        std::cout << (fm.move.isCapture() ? "x" : "-");
        std::cout << square_to_display(fm.path[j], white_to_move);
      }
    }
  }
  std::cout << "\n";
}

static bool parse_move(const Board& board, const std::string& input, bool white_to_move, Move& result) {
  std::vector<FullMove> moves;
  generateFullMoves(board, moves);

  std::vector<int> squares;
  std::string num;
  for (char c : input) {
    if (c >= '0' && c <= '9') {
      num += c;
    } else if (c == '-' || c == 'x' || c == 'X' || c == ':') {
      if (!num.empty()) {
        squares.push_back(display_to_square(std::stoi(num), white_to_move));
        num.clear();
      }
    }
  }
  if (!num.empty()) {
    squares.push_back(display_to_square(std::stoi(num), white_to_move));
  }

  if (squares.size() < 2) return false;

  for (const auto& fm : moves) {
    if (fm.path.size() >= 2 &&
        fm.path[0] == squares[0] &&
        fm.path.back() == squares.back()) {
      if (squares.size() > 2) {
        bool match = (fm.path.size() == squares.size());
        if (match) {
          for (size_t i = 0; i < squares.size(); ++i) {
            if (fm.path[i] != squares[i]) { match = false; break; }
          }
        }
        if (!match) continue;
      }
      result = fm.move;
      return true;
    }
  }
  return false;
}

// ============================================================================
// Engine move
// ============================================================================

static void engine_move(GameState& state, search::Searcher& searcher, Mode mode, bool white_to_move) {
  // Set draw_value based on mode and game ply.
  // Espada: draws = loss for original white. Broquel: draws = win for original white.
  // draw_value is from root (stm) perspective.
  int dv = (mode == Mode::ESPADA) ? -search::SCORE_TB_WIN : search::SCORE_TB_WIN;
  if (state.ply % 2 != 0) dv = -dv;
  searcher.set_draw_value(dv);
  searcher.set_perspective(white_to_move);

  std::cout << "Engine is thinking... (Ctrl+C to move now)\n";

  search::SearchResult result;
  if (state.nodes > 0) {
    result = searcher.search(state.board, 100, search::TimeControl::with_nodes(state.nodes));
  } else {
    auto tc = search::TimeControl::with_time(state.time_limit, state.time_limit * 3);
    result = searcher.search(state.board, 100, tc);
  }

  g_stop_requested.store(false, std::memory_order_relaxed);

  if (result.best_move.from_xor_to == 0) {
    std::cout << "Engine has no move - game over!\n";
    return;
  }

  std::string move_str = move_to_string(state.board, result.best_move, white_to_move);
  int display_score = search::undecided_to_display(result.score);
  std::cout << "Engine plays: " << move_str << " (score: " << display_score << ")\n";

  state.history.push_back(state.board);
  state.board = makeMove(state.board, result.best_move);
  state.ply++;
  ++state.position_counts[state.board.position_hash()];
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " [options]\n"
            << "\nPlay against the engine using proven positions (Espada/Broquel).\n"
            << "\nOptions:\n"
            << "  --mode espada|broquel   Game mode (default: espada)\n"
            << "  --model PATH            Neural network model\n"
            << "  --tb-dir PATH           CWDL tablebase directory (default: /home/alvaro/claude/damas)\n"
            << "  --tb-pieces N           Max pieces for WDL TB (default: 8)\n"
            << "  --known-file PATH       Proven positions file (default: known_<mode>.bin)\n"
            << "  --opening MOVES         Opening moves (e.g. \"1. 10-14 22-18\")\n"
            << "  --nodes N               Search node limit (default: use time)\n"
            << "  --time SECONDS          Search time limit (default: 5.0)\n"
            << "  --tt-size MB            Transposition table size (default: 128)\n"
            << "  --color white|black     Engine plays this color (default: black)\n"
            << "  -h, --help              Show this help\n"
            << "\nCommands during play:\n"
            << "  <move>  Enter a move (e.g., 11-15 or 11x18x25)\n"
            << "  b       Take back the last move\n"
            << "  2       Toggle 2-player mode\n"
            << "  m       Assign engine to current side\n"
            << "  n N     Set search node limit (0 = use time)\n"
            << "  t S     Set search time limit in seconds\n"
            << "  ?       Show legal moves\n"
            << "  q       Quit\n";
}

int main(int argc, char** argv) {
  Mode mode = Mode::ESPADA;
  std::string tb_dir = "/home/alvaro/claude/damas";
  int tb_pieces = 8;
  std::string nn_model;
  std::string known_file;
  std::string opening;
  uint64_t nodes = 0;
  double time_limit = 5.0;
  int tt_size = 128;
  int engine_color = -1;  // -1 = default (black), 0 = white, 1 = black

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--mode" && i + 1 < argc) {
      std::string m = argv[++i];
      if (m == "espada") mode = Mode::ESPADA;
      else if (m == "broquel") mode = Mode::BROQUEL;
      else { std::cerr << "Unknown mode: " << m << "\n"; return 1; }
    } else if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
    } else if (arg == "--tb-dir" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--tb-pieces" && i + 1 < argc) {
      tb_pieces = std::stoi(argv[++i]);
    } else if (arg == "--known-file" && i + 1 < argc) {
      known_file = argv[++i];
    } else if (arg == "--opening" && i + 1 < argc) {
      opening = argv[++i];
    } else if (arg == "--nodes" && i + 1 < argc) {
      nodes = std::stoull(argv[++i]);
    } else if (arg == "--time" && i + 1 < argc) {
      time_limit = std::stod(argv[++i]);
    } else if (arg == "--tt-size" && i + 1 < argc) {
      tt_size = std::stoi(argv[++i]);
    } else if (arg == "--color" && i + 1 < argc) {
      std::string c = argv[++i];
      if (c == "white" || c == "w") engine_color = 0;
      else if (c == "black" || c == "b") engine_color = 1;
      else { std::cerr << "Unknown color: " << c << "\n"; return 1; }
    } else if (arg == "-h" || arg == "--help") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown option: " << arg << "\n";
      return 1;
    }
  }

  if (known_file.empty()) {
    known_file = (mode == Mode::ESPADA) ? "known_espada.bin" : "known_broquel.bin";
  }

  // Load known positions
  KnownPositions known;
  if (known.load(known_file)) {
    std::cout << "Loaded " << known.size() << " proven positions from " << known_file << "\n";
  } else {
    std::cout << "No proven positions loaded (file: " << known_file << ")\n";
  }

  // Load compressed tablebases
  CompressedTablebaseManager cwdl(tb_dir);
  std::cout << "Preloading compressed tablebases from " << tb_dir
            << " (up to " << tb_pieces << " pieces)...\n";
  cwdl.preload(tb_pieces);

  // Set up searcher (no DTM, use CWDL + known positions via WDL probe)
  search::Searcher searcher("", 0, nn_model);
  searcher.set_tt_size(tt_size);
  searcher.set_verbose(true);
  searcher.set_stop_flag(&g_stop_requested);
  std::signal(SIGINT, sigint_handler);

  // Combined WDL probe: known positions + CWDL tables.
  // Known positions are checked at both parities since the search doesn't
  // track game_ply. Board flipping ensures at most one parity matches.
  auto wdl_probe = [&cwdl, &known, tb_pieces](const Board& b) -> std::optional<int> {
    // Check known positions first (stored in stm perspective: +1=win, -1=loss)
    if (b.n_reversible == 0) {
      for (int p = 0; p <= 1; p++) {
        auto r = known.lookup(b, p);
        if (r) return *r;
      }
    }

    // Fall back to CWDL tables
    int pieces = std::popcount(b.allPieces());
    if (pieces <= tb_pieces) {
      Value v = cwdl.lookup_wdl_preloaded(b);
      switch (v) {
        case Value::WIN: return 1;
        case Value::LOSS: return -1;
        case Value::DRAW: return 0;
        default: return std::nullopt;
      }
    }
    return std::nullopt;
  };

  // Set piece limit to 24 so known positions at any piece count are probed
  searcher.set_wdl_probe(wdl_probe, 24);

  // Apply opening moves
  GameState state;
  if (!opening.empty()) {
    GameRecord record = parseGame(Board(), opening);
    if (!record.complete) {
      std::cerr << "Error parsing opening: " << record.error << "\n";
      return 1;
    }
    state.board = record.finalBoard;
    state.ply = static_cast<int>(record.moves.size());
    // Rebuild position counts for the opening
    state.position_counts.clear();
    Board b;
    state.position_counts[b.position_hash()]++;
    for (const auto& move : record.moves) {
      b = makeMove(b, move.move);
      state.position_counts[b.position_hash()]++;
    }
    std::cout << "Opening (" << state.ply << " plies): " << opening << "\n";
  }
  state.nodes = nodes;
  state.time_limit = time_limit;

  // Set engine color
  if (engine_color == 0) {
    state.engine_plays_white = true;
    state.engine_plays_black = false;
  } else if (engine_color == 1) {
    state.engine_plays_white = false;
    state.engine_plays_black = true;
  }
  // else keep defaults (engine plays black)

  const char* mode_name = (mode == Mode::ESPADA) ? "Espada" : "Broquel";
  std::cout << "\n=== " << mode_name << " Play ===\n";
  if (mode == Mode::ESPADA) {
    std::cout << "Draws count as wins for Black.\n";
  } else {
    std::cout << "Draws count as wins for White.\n";
  }
  std::cout << "Engine plays " << (state.engine_plays_white ? "White" : "Black")
            << ", you play " << (state.engine_plays_white ? "Black" : "White") << ".\n";
  std::cout << "Enter moves like '11-15' or '11x18'. Type '?' for help.\n";

  while (true) {
    bool white_to_move = (state.ply % 2 == 0);
    print_board(state.board, white_to_move);

    // Check for game over
    MoveList moves;
    generateMoves(state.board, moves);
    if (moves.empty()) {
      std::cout << (white_to_move ? "White" : "Black") << " has no moves - "
                << (white_to_move ? "Black" : "White") << " wins!\n";
      break;
    }

    // Check for draw by repetition
    if (state.position_counts[state.board.position_hash()] >= 3) {
      std::cout << "Draw by threefold repetition!\n";
      if (mode == Mode::ESPADA)
        std::cout << "In Espada: this counts as a win for Black.\n";
      else
        std::cout << "In Broquel: this counts as a win for White.\n";
      break;
    }

    // Check for draw by 60 reversible moves
    if (state.board.n_reversible >= 60) {
      std::cout << "Draw by 60-move rule!\n";
      if (mode == Mode::ESPADA)
        std::cout << "In Espada: this counts as a win for Black.\n";
      else
        std::cout << "In Broquel: this counts as a win for White.\n";
      break;
    }

    // Check if engine should play
    bool engine_turn = (!state.two_player_mode) &&
                       ((white_to_move && state.engine_plays_white) ||
                        (!white_to_move && state.engine_plays_black));

    if (engine_turn) {
      engine_move(state, searcher, mode, white_to_move);
      continue;
    }

    // Get user input
    std::cout << "> ";
    std::string input;
    if (!std::getline(std::cin, input)) break;

    // Trim
    while (!input.empty() && (input.back() == ' ' || input.back() == '\n' || input.back() == '\r'))
      input.pop_back();
    while (!input.empty() && input.front() == ' ')
      input.erase(input.begin());
    if (input.empty()) continue;

    if (input == "q" || input == "quit" || input == "exit") {
      break;
    } else if (input == "?" || input == "help") {
      show_legal_moves(state.board, white_to_move);
      std::cout << "\nCommands: b=back, 2=two-player, m=engine takes this side,"
                << " n N=nodes, t S=time, q=quit\n";
    } else if (input == "b" || input == "back" || input == "undo") {
      if (state.history.empty()) {
        std::cout << "Nothing to undo!\n";
      } else {
        auto it = state.position_counts.find(state.board.position_hash());
        if (it != state.position_counts.end() && it->second > 0) --it->second;
        state.board = state.history.back();
        state.history.pop_back();
        state.ply--;
        std::swap(state.engine_plays_white, state.engine_plays_black);
        std::cout << "Move taken back.\n";
      }
    } else if (input == "2") {
      state.two_player_mode = !state.two_player_mode;
      std::cout << "Two-player mode: " << (state.two_player_mode ? "ON" : "OFF") << "\n";
    } else if (input == "m" || input == "move") {
      state.engine_plays_white = white_to_move;
      state.engine_plays_black = !white_to_move;
      state.two_player_mode = false;
      std::cout << "Engine now plays " << (white_to_move ? "White" : "Black")
                << ", you play " << (white_to_move ? "Black" : "White") << "\n";
    } else if (input.size() >= 2 && input[0] == 'n' && input[1] == ' ') {
      state.nodes = std::stoull(input.substr(2));
      if (state.nodes == 0)
        std::cout << "Using time control (" << state.time_limit << "s)\n";
      else
        std::cout << "Node limit set to " << state.nodes << "\n";
    } else if (input.size() >= 2 && input[0] == 't' && input[1] == ' ') {
      state.time_limit = std::stod(input.substr(2));
      state.nodes = 0;
      std::cout << "Time limit set to " << state.time_limit << "s\n";
    } else {
      Move move;
      if (parse_move(state.board, input, white_to_move, move)) {
        std::cout << "Playing: " << move_to_string(state.board, move, white_to_move) << "\n";
        state.history.push_back(state.board);
        state.board = makeMove(state.board, move);
        state.ply++;
        ++state.position_counts[state.board.position_hash()];
      } else {
        std::cout << "Invalid move or command. Type '?' for legal moves.\n";
      }
    }
  }

  std::cout << "Thanks for playing!\n";
  return 0;
}
