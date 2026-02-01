// Interactive play against the engine
//
// Commands:
//   <move>  - Enter a move (e.g., "11-15" or "11x18x25")
//   b       - Take back the last move
//   2       - Toggle 2-player mode (engine doesn't play)
//   m       - Assign engine to current side (you take the other)
//   n N     - Set search node limit
//   q       - Quit
//   ?       - Show legal moves

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "search/search.hpp"
#include "tablebase/tb_probe.hpp"
#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <vector>

// Global stop flag for SIGINT handling
std::atomic<bool> g_stop_requested{false};

void sigint_handler(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

// Game state
struct GameState {
  Board board;
  std::vector<Board> history;  // For undo
  bool engine_plays_white = false;
  bool engine_plays_black = true;  // Engine plays black by default
  bool two_player_mode = false;
  int ply = 0;
  std::uint64_t nodes = 100000;  // Search node limit
};

// Print the board with side to move
void print_board(const Board& board, bool white_to_move) {
  // Board is stored with side-to-move as white, flip for display if it's black's turn
  Board display = white_to_move ? board : flip(board);
  std::cout << "\n" << display;
  std::cout << (white_to_move ? "White" : "Black") << " to move\n";
}

// Convert square index to display number (flip for black's perspective)
int square_to_display(int sq, bool white_to_move) {
  int display = sq + 1;  // 0-indexed to 1-indexed
  return white_to_move ? display : (33 - display);
}

// Get move notation for a move
std::string move_to_string(const Board& board, const Move& move, bool white_to_move) {
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

// Show legal moves
void show_legal_moves(const Board& board, bool white_to_move) {
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

// Convert display number to square index (flip for black's perspective)
int display_to_square(int display, bool white_to_move) {
  int sq = white_to_move ? display : (33 - display);
  return sq - 1;  // 1-indexed to 0-indexed
}

// Parse a move from user input
// Returns true if move was found, sets result
bool parse_move(const Board& board, const std::string& input, bool white_to_move, Move& result) {
  std::vector<FullMove> moves;
  generateFullMoves(board, moves);

  // Parse input like "11-15" or "11x18x25"
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

  if (squares.size() < 2) {
    return false;
  }

  // Find matching move
  for (const auto& fm : moves) {
    if (fm.path.size() >= 2 &&
        fm.path[0] == squares[0] &&
        fm.path.back() == squares.back()) {
      // Check if path matches for multi-jump
      if (squares.size() > 2) {
        bool match = (fm.path.size() == squares.size());
        if (match) {
          for (size_t i = 0; i < squares.size(); ++i) {
            if (fm.path[i] != squares[i]) {
              match = false;
              break;
            }
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

// Make the engine play
void engine_move(GameState& state, search::Searcher& searcher, bool white_to_move) {
  std::cout << "Engine is thinking... (Ctrl+C to move now)\n";

  searcher.set_perspective(white_to_move);
  searcher.set_root_white_to_move(white_to_move);
  auto result = searcher.search_nodes(state.board, state.nodes);

  // Reset stop flag if it was set
  g_stop_requested.store(false, std::memory_order_relaxed);

  if (result.best_move.from_xor_to == 0) {
    std::cout << "Engine has no move - game over!\n";
    return;
  }

  std::string move_str = move_to_string(state.board, result.best_move, white_to_move);
  std::cout << "Engine plays: " << move_str << " (score: " << result.score << ")\n";

  state.history.push_back(state.board);
  state.board = makeMove(state.board, result.best_move);
  state.ply++;
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string nn_model = "";
  std::string dtm_nn_model = "";
  std::uint64_t nodes = 100000;
  bool use_tb = true;
  bool use_dtm_tb = true;  // Use DTM tablebases for optimal play
  int tb_limit = 7;  // Use tablebases for positions with this many pieces or fewer
  int draw_score = 0;  // Value of a draw from white's perspective

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
    } else if (arg == "--dtm-model" && i + 1 < argc) {
      dtm_nn_model = argv[++i];
    } else if (arg == "--tb" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--no-tb") {
      use_tb = false;
    } else if (arg == "--no-dtm-tb") {
      use_dtm_tb = false;
    } else if (arg == "--tb-limit" && i + 1 < argc) {
      tb_limit = std::stoi(argv[++i]);
    } else if (arg == "--nodes" && i + 1 < argc) {
      nodes = std::stoull(argv[++i]);
    } else if (arg == "--draw-score" && i + 1 < argc) {
      draw_score = std::stoi(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --model FILE      Neural network model (general eval)\n"
                << "  --dtm-model FILE  DTM specialist model (endgame eval)\n"
                << "  --tb PATH         Tablebase directory\n"
                << "  --tb-limit N      Use tablebases for N pieces or fewer (default: 7)\n"
                << "  --no-tb           Disable all tablebases (use NN only)\n"
                << "  --no-dtm-tb       Disable DTM tablebases (use only cwdl_*.bin)\n"
                << "  --nodes N         Search node limit (default: 100000)\n"
                << "  --draw-score N    Value of draw for white (default: 0, use -100 for aggression)\n"
                << "\nCommands during play:\n"
                << "  <move>  Enter a move (e.g., 11-15 or 11x18x25)\n"
                << "  b       Take back the last move\n"
                << "  2       Toggle 2-player mode\n"
                << "  m       Assign engine to current side (you take the other)\n"
                << "  n N     Set search node limit\n"
                << "  ?       Show legal moves\n"
                << "  q       Quit\n";
      return 0;
    }
  }

  // Initialize engine
  std::cout << "Initializing engine...\n";
  if (use_tb) {
    std::cout << "  WDL tablebases: " << tb_dir << " (limit: " << tb_limit << " pieces)\n";
  } else {
    std::cout << "  WDL tablebases: disabled\n";
    tb_dir = "";  // Empty disables TB loading
    tb_limit = 0;
  }

  // dtm_limit controls DTM optimal play (usually tb_limit - 1)
  int dtm_limit = use_dtm_tb ? std::max(0, tb_limit - 1) : 0;
  if (use_tb && use_dtm_tb) {
    std::cout << "  DTM tablebases: enabled (limit: " << dtm_limit << " pieces)\n";
  } else {
    std::cout << "  DTM tablebases: disabled\n";
  }

  if (!nn_model.empty()) {
    std::cout << "  General model: " << nn_model << "\n";
  }
  if (!dtm_nn_model.empty()) {
    std::cout << "  DTM NN model: " << dtm_nn_model << "\n";
  }

  search::Searcher searcher(tb_dir, tb_limit, dtm_limit, nn_model, dtm_nn_model);
  searcher.set_tt_size(2048);
  searcher.set_verbose(true);
  searcher.set_stop_flag(&g_stop_requested);
  std::signal(SIGINT, sigint_handler);
  if (draw_score != 0) {
    searcher.set_draw_score(draw_score);
    std::cout << "  Draw score: " << draw_score << " (for white)\n";
  }

  GameState state;
  state.nodes = nodes;

  std::cout << "\n=== Checkers ===\n";
  std::cout << "You play White (w/W), engine plays Black (b/B)\n";
  std::cout << "Enter moves like '11-15' or '11x18'. Type '?' for help.\n";

  while (true) {
    // Determine whose turn it is from game perspective
    bool white_to_move = (state.ply % 2 == 0);

    print_board(state.board, white_to_move);

    // Check for game over
    MoveList moves;
    generateMoves(state.board, moves);
    if (moves.empty()) {
      std::cout << (white_to_move ? "White" : "Black") << " has no moves - ";
      std::cout << (white_to_move ? "Black" : "White") << " wins!\n";
      break;
    }

    // Check if engine should play
    bool engine_turn = (!state.two_player_mode) &&
                       ((white_to_move && state.engine_plays_white) ||
                        (!white_to_move && state.engine_plays_black));

    if (engine_turn) {
      engine_move(state, searcher, white_to_move);
      continue;
    }

    // Get user input
    std::cout << "> ";
    std::string input;
    if (!std::getline(std::cin, input)) {
      break;
    }

    // Trim whitespace
    while (!input.empty() && (input.back() == ' ' || input.back() == '\n' || input.back() == '\r')) {
      input.pop_back();
    }
    while (!input.empty() && input.front() == ' ') {
      input.erase(input.begin());
    }

    if (input.empty()) continue;

    // Process commands
    if (input == "q" || input == "quit" || input == "exit") {
      break;
    } else if (input == "?" || input == "help") {
      show_legal_moves(state.board, white_to_move);
      std::cout << "\nCommands: b=back, 2=two-player, m=engine takes this side, n N=nodes, q=quit\n";
    } else if (input == "b" || input == "back" || input == "undo") {
      if (state.history.empty()) {
        std::cout << "Nothing to undo!\n";
      } else {
        state.board = state.history.back();
        state.history.pop_back();
        state.ply--;
        // Swap assigned players
        std::swap(state.engine_plays_white, state.engine_plays_black);
        std::cout << "Move taken back.\n";
      }
    } else if (input == "2") {
      state.two_player_mode = !state.two_player_mode;
      std::cout << "Two-player mode: " << (state.two_player_mode ? "ON" : "OFF") << "\n";
    } else if (input == "m" || input == "move") {
      // Assign engine to current side, human to the other
      state.engine_plays_white = white_to_move;
      state.engine_plays_black = !white_to_move;
      state.two_player_mode = false;
      std::cout << "Engine now plays " << (white_to_move ? "White" : "Black")
                << ", you play " << (white_to_move ? "Black" : "White") << "\n";
    } else if (input.size() >= 2 && input[0] == 'n' && input[1] == ' ') {
      // Set node limit
      state.nodes = std::stoull(input.substr(2));
      std::cout << "Node limit set to " << state.nodes << "\n";
    } else {
      // Try to parse as a move
      Move move;
      if (parse_move(state.board, input, white_to_move, move)) {
        std::cout << "Playing: " << move_to_string(state.board, move, white_to_move) << "\n";
        state.history.push_back(state.board);
        state.board = makeMove(state.board, move);
        state.ply++;
      } else {
        std::cout << "Invalid move or command. Type '?' for legal moves.\n";
      }
    }
  }

  std::cout << "Thanks for playing!\n";
  return 0;
}
