// Opening book generator for white
//
// Iteratively builds a book mapping positions to probability distributions
// over legal moves. Each iteration walks existing book moves, then expands
// at leaf nodes with deep searches.

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tb_probe.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Global stop flag for SIGINT handling
std::atomic<bool> g_stop_requested{false};

void sigint_handler(int) {
  g_stop_requested.store(true, std::memory_order_relaxed);
}

struct BookEntry {
  // For file I/O and debugging: the board stored with white-to-move
  Board board;
  std::vector<Move> moves;
  std::vector<double> weights;  // normalized to sum to 1.0
};

using Book = std::unordered_map<uint64_t, BookEntry>;

// Save book to file
void save_book(const Book& book, const std::string& filename) {
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Error: cannot write to " << filename << "\n";
    return;
  }

  out << "# White opening book\n";
  out << "# position: white_hex black_hex kings_hex\n";
  out << "# move: from_xor_to_hex captures_hex weight\n";
  out << std::fixed << std::setprecision(6);

  for (const auto& [hash, entry] : book) {
    out << "\nP " << std::hex
        << entry.board.white << " "
        << entry.board.black << " "
        << entry.board.kings
        << std::dec << "\n";
    for (size_t i = 0; i < entry.moves.size(); ++i) {
      out << "M " << std::hex
          << entry.moves[i].from_xor_to << " "
          << entry.moves[i].captures
          << std::dec << " "
          << entry.weights[i] << "\n";
    }
  }

  out.close();
  std::cout << "Book saved to " << filename << " (" << book.size() << " positions)\n";
}

// Load book from file
void load_book(Book& book, const std::string& filename) {
  std::ifstream in(filename);
  if (!in) return;  // File doesn't exist yet, start fresh

  std::string line;
  Board current_board;
  uint64_t current_hash = 0;
  bool have_position = false;

  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;

    if (line[0] == 'P') {
      // Parse position
      Bb w, b, k;
      if (sscanf(line.c_str(), "P %x %x %x", &w, &b, &k) == 3) {
        current_board = Board(w, b, k);
        current_hash = current_board.position_hash();
        have_position = true;
        book[current_hash].board = current_board;
        book[current_hash].moves.clear();
        book[current_hash].weights.clear();
      }
    } else if (line[0] == 'M' && have_position) {
      // Parse move
      Bb fxt, cap;
      double weight;
      if (sscanf(line.c_str(), "M %x %x %lf", &fxt, &cap, &weight) == 3) {
        book[current_hash].moves.push_back(Move(fxt, cap));
        book[current_hash].weights.push_back(weight);
      }
    }
  }

  std::cout << "Loaded book from " << filename << " (" << book.size() << " positions)\n";
}

// Sample a move index from a distribution using the given RNG
int sample_move(const std::vector<double>& weights, RandomBits& rng) {
  double r = (rng() >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
  double cumulative = 0;
  for (size_t i = 0; i < weights.size(); ++i) {
    cumulative += weights[i];
    if (r < cumulative) return static_cast<int>(i);
  }
  return static_cast<int>(weights.size() - 1);
}

// Get move notation for display
// When black_perspective is true, flips square numbers so they display from black's view
std::string move_notation(const Board& board, const Move& move, bool black_perspective = false) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    if (fm.move == move) {
      if (!black_perspective) return moveToString(fm);
      // Flip square numbers: internal sq N → game sq (32 - N)
      std::ostringstream oss;
      char sep = fm.move.isCapture() ? 'x' : '-';
      for (size_t i = 0; i < fm.path.size(); ++i) {
        if (i > 0) oss << sep;
        oss << (32 - fm.path[i]);
      }
      return oss.str();
    }
  }
  // Fallback: hex representation
  std::ostringstream oss;
  oss << std::hex << move.from_xor_to;
  if (move.captures) oss << "x" << move.captures;
  return oss.str();
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string nn_model;
  std::string dtm_nn_model;
  std::string book_file = "white.book";
  uint64_t white_nodes = 100000000;
  uint64_t black_nodes = 1000000;
  int threshold = 200;
  int black_threshold = 2000;
  double white_temperature = 100.0;
  double black_temperature = 1000.0;
  int max_ply = 30;
  int max_iterations = 0;  // 0 = unlimited
  int tb_limit = 7;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
    } else if (arg == "--dtm-model" && i + 1 < argc) {
      dtm_nn_model = argv[++i];
    } else if (arg == "--tb" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--no-tb") {
      tb_dir = "";
      tb_limit = 0;
    } else if (arg == "--white-nodes" && i + 1 < argc) {
      white_nodes = std::stoull(argv[++i]);
    } else if (arg == "--black-nodes" && i + 1 < argc) {
      black_nodes = std::stoull(argv[++i]);
    } else if (arg == "--threshold" && i + 1 < argc) {
      threshold = std::stoi(argv[++i]);
    } else if (arg == "--black-threshold" && i + 1 < argc) {
      black_threshold = std::stoi(argv[++i]);
    } else if (arg == "--white-temperature" && i + 1 < argc) {
      white_temperature = std::stod(argv[++i]);
    } else if (arg == "--black-temperature" && i + 1 < argc) {
      black_temperature = std::stod(argv[++i]);
    } else if (arg == "--max-ply" && i + 1 < argc) {
      max_ply = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      max_iterations = std::stoi(argv[++i]);
    } else if (arg == "--book" && i + 1 < argc) {
      book_file = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Generate a white opening book\n\n"
                << "Options:\n"
                << "  --model FILE         Neural network model\n"
                << "  --dtm-model FILE     DTM specialist model\n"
                << "  --tb PATH            Tablebase directory (default: /home/alvaro/claude/damas)\n"
                << "  --no-tb              Disable tablebases\n"
                << "  --white-nodes N      Node limit for white moves (default: 100000000)\n"
                << "  --black-nodes N      Node limit for black moves (default: 1000000)\n"
                << "  --threshold N        Score threshold for white moves (default: 200)\n"
                << "  --black-threshold N  Score threshold for black moves (default: 2000)\n"
                << "  --white-temperature T  Softmax temperature for white (default: 100)\n"
                << "  --black-temperature T  Softmax temperature for black (default: 1000)\n"
                << "  --max-ply N          Maximum book depth in plies (default: 30)\n"
                << "  --iterations N       Number of iterations (default: unlimited)\n"
                << "  --book FILE          Book file to load/save (default: white.book)\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  // Initialize
  std::cout << "=== Opening Book Generator ===\n";
  std::cout << "White nodes: " << white_nodes << "\n";
  std::cout << "Black nodes: " << black_nodes << "\n";
  std::cout << "White: threshold=" << threshold << " temperature=" << white_temperature << "\n";
  std::cout << "Black: threshold=" << black_threshold << " temperature=" << black_temperature << "\n";
  std::cout << "Max ply: " << max_ply << "\n";
  std::cout << "Book file: " << book_file << "\n";

  if (!nn_model.empty()) std::cout << "Model: " << nn_model << "\n";
  if (!dtm_nn_model.empty()) std::cout << "DTM model: " << dtm_nn_model << "\n";
  if (!tb_dir.empty()) std::cout << "Tablebases: " << tb_dir << "\n";

  search::Searcher searcher(tb_dir, tb_limit, nn_model, dtm_nn_model);
  searcher.set_tt_size(64);
  searcher.set_verbose(false);
  searcher.set_draw_score(-2000);
  searcher.set_stop_flag(&g_stop_requested);

  std::signal(SIGINT, sigint_handler);

  // RNG for sampling
  auto seed = static_cast<uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  RandomBits rng(seed);

  // Load existing book
  Book book;
  load_book(book, book_file);

  int iteration = 0;

  while (max_iterations == 0 || iteration < max_iterations) {
    if (g_stop_requested.load(std::memory_order_relaxed)) break;

    iteration++;
    std::cout << "\n--- Iteration " << iteration << " ---\n";

    // Walk the book from the initial position
    Board board;  // initial position
    int ply = 0;
    bool expanded = false;

    while (ply < max_ply) {
      if (g_stop_requested.load(std::memory_order_relaxed)) break;

      // Check for game over
      MoveList legal_moves;
      generateMoves(board, legal_moves);
      if (legal_moves.empty()) {
        std::cout << "  Game over at ply " << ply << "\n";
        break;
      }

      // Check if position is in book
      uint64_t hash = board.position_hash();
      auto it = book.find(hash);

      if (it != book.end()) {
        // Position is in book — sample a move and continue
        int idx = sample_move(it->second.weights, rng);
        Move move = it->second.moves[idx];
        bool white_to_move = (ply % 2 == 0);
        std::string notation = move_notation(board, move, !white_to_move);
        std::cout << "  Ply " << ply << " (" << (white_to_move ? "W" : "B")
                  << "): book move " << notation << "\n";
        board = makeMove(board, move);
        ply++;
        continue;
      }

      // Position not in book — expand it
      bool white_to_move = (ply % 2 == 0);
      std::cout << "  Expanding at ply " << ply << " (" << (white_to_move ? "White" : "Black")
                << " to move), " << legal_moves.size() << " legal moves\n";
      Board display = white_to_move ? board : flip(board);
      std::cout << display;

      // Search the position once, getting scores for all moves at the same depth
      uint64_t node_limit = white_to_move ? white_nodes : black_nodes;
      int search_threshold = white_to_move ? threshold : black_threshold;
      searcher.set_root_white_to_move(white_to_move);
      searcher.set_perspective(white_to_move);
      searcher.clear_tt();

      auto result = searcher.search_multi(board, 100, node_limit, search_threshold);

      if (g_stop_requested.load(std::memory_order_relaxed)) break;
      if (result.moves.empty()) break;

      // Print all move scores (result is sorted by score descending)
      int best_score = result.moves[0].score;
      for (const auto& ms : result.moves) {
        std::string notation = move_notation(board, ms.move, !white_to_move);
        std::cout << "    " << notation << ": " << ms.score << "\n";
      }
      std::cout << "  depth " << result.depth << ", " << result.nodes << " nodes\n";

      // Build book entry
      BookEntry entry;
      entry.board = board;

      int thresh = white_to_move ? threshold : black_threshold;
      double temp = white_to_move ? white_temperature : black_temperature;

      for (const auto& ms : result.moves) {
        if (ms.score >= best_score - thresh) {
          entry.moves.push_back(ms.move);
          double w = std::exp((ms.score - best_score) / temp);
          entry.weights.push_back(w);
        }
      }

      // Normalize weights
      double total = 0;
      for (double w : entry.weights) total += w;
      if (total > 0) {
        for (double& w : entry.weights) w /= total;
      }

      // Print selected moves
      std::cout << "  Selected " << entry.moves.size() << " moves (best score: " << best_score << "):\n";
      for (size_t i = 0; i < entry.moves.size(); ++i) {
        std::string notation = move_notation(board, entry.moves[i], !white_to_move);
        std::cout << "    " << notation << " weight=" << std::fixed << std::setprecision(3)
                  << entry.weights[i] << "\n";
      }

      book[hash] = std::move(entry);
      expanded = true;

      // Save after each expansion
      save_book(book, book_file);

      if (white_to_move) {
        // Expanded a white node — done with this iteration
        break;
      }

      // Expanded a black node — sample a move and keep walking to find a white leaf
      auto& new_entry = book[hash];
      int idx = sample_move(new_entry.weights, rng);
      board = makeMove(board, new_entry.moves[idx]);
      ply++;
    }

    if (g_stop_requested.load(std::memory_order_relaxed)) break;

    if (!expanded) {
      std::cout << "  No expansion this iteration (reached max ply or game over)\n";
    }
  }

  // Final save
  save_book(book, book_file);
  std::cout << "\nDone. Book has " << book.size() << " positions.\n";
  return 0;
}
