// PUCT-based opening book generator
//
// Uses a PUCT (Predictor + Upper Confidence bound for Trees) algorithm inspired
// by AlphaZero's MCTS. Each iteration walks from the root, selecting moves via
// PUCT, expands a leaf node, and backpropagates the evaluation.
//
// Key properties:
// - All legal moves are always candidates (no threshold-based pruning)
// - PUCT formula balances exploitation (high Q) vs exploration (high U)
// - Values improve over time via backpropagation

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
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

struct MoveInfo {
  Move move;
  int visits;        // times this move was taken (starts at 1 for virtual visit)
  double value_sum;  // sum of evaluations (from side-to-move perspective)
  double prior;      // softmax prior from initial expansion search
};

struct BookEntry {
  Board board;
  std::vector<MoveInfo> moves;  // ALL legal moves
};

using Book = std::unordered_map<uint64_t, BookEntry>;

// Save book to file (new PUCT format)
void save_book(const Book& book, const std::string& filename) {
  std::ofstream out(filename);
  if (!out) {
    std::cerr << "Error: cannot write to " << filename << "\n";
    return;
  }

  out << "# PUCT opening book v1\n";
  out << std::fixed << std::setprecision(6);

  for (const auto& [hash, entry] : book) {
    out << "\nP " << std::hex
        << entry.board.white << " "
        << entry.board.black << " "
        << entry.board.kings
        << std::dec << "\n";
    for (const auto& mi : entry.moves) {
      out << "M " << std::hex
          << mi.move.from_xor_to << " "
          << mi.move.captures
          << std::dec << " "
          << mi.visits << " "
          << mi.value_sum << " "
          << mi.prior << "\n";
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
      Bb w, b, k;
      if (sscanf(line.c_str(), "P %x %x %x", &w, &b, &k) == 3) {
        current_board = Board(w, b, k);
        current_hash = current_board.position_hash();
        have_position = true;
        book[current_hash].board = current_board;
        book[current_hash].moves.clear();
      }
    } else if (line[0] == 'M' && have_position) {
      Bb fxt, cap;
      int visits;
      double value_sum, prior;
      if (sscanf(line.c_str(), "M %x %x %d %lf %lf", &fxt, &cap, &visits, &value_sum, &prior) == 5) {
        book[current_hash].moves.push_back({Move(fxt, cap), visits, value_sum, prior});
      }
    }
  }

  std::cout << "Loaded book from " << filename << " (" << book.size() << " positions)\n";
}

// Get move notation for display
std::string move_notation(const Board& board, const Move& move, bool black_perspective = false) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    if (fm.move == move) {
      if (!black_perspective) return moveToString(fm);
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

// PUCT move selection: returns index of the selected move
int select_puct(const BookEntry& entry, double c_puct) {
  int total_visits = 0;
  for (const auto& mi : entry.moves) {
    total_visits += mi.visits;
  }
  double sqrt_total = std::sqrt(static_cast<double>(total_visits));

  int best_idx = 0;
  double best_score = -1e18;
  for (size_t i = 0; i < entry.moves.size(); ++i) {
    const auto& mi = entry.moves[i];
    double q = mi.value_sum / mi.visits;
    double u = c_puct * mi.prior * sqrt_total / (1 + mi.visits);
    double score = q + u;
    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string nn_model;
  std::string dtm_nn_model;
  std::string book_file = "opening.book";
  uint64_t nodes = 10000000;
  double c_puct = 1000.0;
  double prior_temp = 1000.0;
  int max_ply = 30;
  int max_iterations = 0;  // 0 = unlimited
  int save_interval = 10;
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
    } else if (arg == "--nodes" && i + 1 < argc) {
      nodes = std::stoull(argv[++i]);
    } else if (arg == "--cpuct" && i + 1 < argc) {
      c_puct = std::stod(argv[++i]);
    } else if (arg == "--prior-temp" && i + 1 < argc) {
      prior_temp = std::stod(argv[++i]);
    } else if (arg == "--max-ply" && i + 1 < argc) {
      max_ply = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      max_iterations = std::stoi(argv[++i]);
    } else if (arg == "--save-interval" && i + 1 < argc) {
      save_interval = std::stoi(argv[++i]);
    } else if (arg == "--book" && i + 1 < argc) {
      book_file = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Generate an opening book using PUCT-based exploration\n\n"
                << "Options:\n"
                << "  --model FILE       Neural network model\n"
                << "  --dtm-model FILE   DTM specialist model\n"
                << "  --tb PATH          Tablebase directory (default: /home/alvaro/claude/damas)\n"
                << "  --no-tb            Disable tablebases\n"
                << "  --nodes N          Node limit for search (default: 10000000)\n"
                << "  --cpuct N          PUCT exploration constant (default: 1000)\n"
                << "  --prior-temp T     Softmax temperature for priors (default: 1000)\n"
                << "  --max-ply N        Maximum book depth in plies (default: 30)\n"
                << "  --iterations N     Number of iterations (default: unlimited)\n"
                << "  --save-interval N  Save every N iterations (default: 10)\n"
                << "  --book FILE        Book file to load/save (default: opening.book)\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  // Initialize
  std::cout << "=== PUCT Opening Book Generator ===\n";
  std::cout << "Nodes: " << nodes << "\n";
  std::cout << "C_PUCT: " << c_puct << "\n";
  std::cout << "Prior temperature: " << prior_temp << "\n";
  std::cout << "Max ply: " << max_ply << "\n";
  std::cout << "Save interval: " << save_interval << "\n";
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

  // Load existing book
  Book book;
  load_book(book, book_file);

  int iteration = 0;
  int unsaved_iterations = 0;

  while (max_iterations == 0 || iteration < max_iterations) {
    if (g_stop_requested.load(std::memory_order_relaxed)) break;

    iteration++;
    std::cout << "\n--- Iteration " << iteration << " ---\n";

    // Path through the tree for backpropagation
    // Each entry: (position hash, move index chosen at that position)
    std::vector<std::pair<uint64_t, int>> path;

    Board current_board;  // initial position
    int ply = 0;
    double leaf_value = 0;
    bool got_leaf = false;

    while (ply < max_ply) {
      if (g_stop_requested.load(std::memory_order_relaxed)) break;

      // Check for game over
      MoveList legal_moves;
      generateMoves(current_board, legal_moves);
      if (legal_moves.empty()) {
        // Side to move loses (no moves = loss)
        leaf_value = -100000;
        got_leaf = true;
        std::cout << "  Game over at ply " << ply << " (no moves)\n";
        break;
      }

      uint64_t hash = current_board.position_hash();
      auto it = book.find(hash);

      if (it != book.end()) {
        // Position is in book — select move by PUCT
        int idx = select_puct(it->second, c_puct);
        const auto& mi = it->second.moves[idx];
        bool white_to_move = (ply % 2 == 0);
        std::string notation = move_notation(current_board, mi.move, !white_to_move);
        double q = mi.value_sum / mi.visits;
        std::cout << "  Ply " << ply << " (" << (white_to_move ? "W" : "B")
                  << "): PUCT " << notation
                  << " (Q=" << std::fixed << std::setprecision(1) << q
                  << " prior=" << std::setprecision(3) << mi.prior
                  << " visits=" << mi.visits << ")\n";

        path.push_back({hash, idx});
        current_board = makeMove(current_board, mi.move);
        ply++;
        continue;
      }

      // Position not in book — expand it
      bool white_to_move = (ply % 2 == 0);

      // Forced move: add to book and keep walking (not a real decision point)
      if (legal_moves.size() == 1) {
        std::string notation = move_notation(current_board, legal_moves[0], !white_to_move);
        std::cout << "  Ply " << ply << " (" << (white_to_move ? "W" : "B")
                  << "): forced " << notation << "\n";

        BookEntry entry;
        entry.board = current_board;
        entry.moves.push_back({legal_moves[0], 1, 0.0, 1.0});
        book[hash] = std::move(entry);

        path.push_back({hash, 0});
        current_board = makeMove(current_board, legal_moves[0]);
        ply++;
        continue;
      }

      std::cout << "  Expanding at ply " << ply << " (" << (white_to_move ? "White" : "Black")
                << " to move), " << legal_moves.size() << " legal moves\n";
      Board display = white_to_move ? current_board : flip(current_board);
      std::cout << display;

      // Search with large threshold to get scores for all moves
      searcher.set_root_white_to_move(white_to_move);
      searcher.set_perspective(white_to_move);
      searcher.clear_tt();

      auto result = searcher.search_multi(current_board, 100, nodes, 30000);

      if (g_stop_requested.load(std::memory_order_relaxed)) break;
      if (result.moves.empty()) break;

      int best_score = result.moves[0].score;

      // Print all move scores
      for (const auto& ms : result.moves) {
        std::string notation = move_notation(current_board, ms.move, !white_to_move);
        std::cout << "    " << notation << ": " << ms.score << "\n";
      }
      std::cout << "  depth " << result.depth << ", " << result.nodes << " nodes\n";

      // Compute softmax priors
      double max_exp = 0;
      std::vector<double> priors(result.moves.size());
      for (size_t i = 0; i < result.moves.size(); ++i) {
        priors[i] = std::exp((result.moves[i].score - best_score) / prior_temp);
        max_exp += priors[i];
      }
      for (auto& p : priors) p /= max_exp;

      // Build book entry with all moves
      BookEntry entry;
      entry.board = current_board;
      for (size_t i = 0; i < result.moves.size(); ++i) {
        entry.moves.push_back({
          result.moves[i].move,
          1,                                        // virtual visit
          static_cast<double>(result.moves[i].score),  // initial value_sum
          priors[i]
        });
      }

      // Print moves with priors
      std::cout << "  " << entry.moves.size() << " moves (best score: " << best_score << "):\n";
      for (const auto& mi : entry.moves) {
        std::string notation = move_notation(current_board, mi.move, !white_to_move);
        std::cout << "    " << notation
                  << " prior=" << std::fixed << std::setprecision(3) << mi.prior
                  << " Q=" << std::setprecision(1) << mi.value_sum << "\n";
      }

      book[hash] = std::move(entry);

      // Leaf value = best score (from side-to-move perspective)
      leaf_value = best_score;
      got_leaf = true;
      break;
    }

    // If we reached max_ply without expanding
    if (!got_leaf && !g_stop_requested.load(std::memory_order_relaxed)) {
      uint64_t hash = current_board.position_hash();
      auto it = book.find(hash);

      if (it != book.end()) {
        // Use best Q as leaf value
        double best_q = -1e18;
        for (const auto& mi : it->second.moves) {
          double q = mi.value_sum / mi.visits;
          if (q > best_q) best_q = q;
        }
        leaf_value = best_q;
        got_leaf = true;
        std::cout << "  Reached max ply (in book), leaf Q=" << std::fixed << std::setprecision(1) << leaf_value << "\n";
      } else {
        // Search without adding to book
        MoveList legal_moves;
        generateMoves(current_board, legal_moves);
        if (!legal_moves.empty()) {
          bool white_to_move = (ply % 2 == 0);
          searcher.set_root_white_to_move(white_to_move);
          searcher.set_perspective(white_to_move);
          searcher.clear_tt();

          auto result = searcher.search_multi(current_board, 100, nodes, 30000);
          if (!result.moves.empty()) {
            leaf_value = result.moves[0].score;
            got_leaf = true;
            std::cout << "  Reached max ply (searched), leaf=" << leaf_value << "\n";
          }
        } else {
          leaf_value = -100000;
          got_leaf = true;
          std::cout << "  Reached max ply (game over)\n";
        }
      }
    }

    // Backpropagate leaf value up the path
    if (got_leaf && !path.empty()) {
      double value = leaf_value;
      std::cout << "  Backpropagating value=" << std::fixed << std::setprecision(1) << value
                << " through " << path.size() << " plies\n";

      for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        value = -value;  // negate (flip perspective)
        auto& entry = book[path[i].first];
        auto& mi = entry.moves[path[i].second];
        mi.value_sum += value;
        mi.visits++;
      }
    }

    if (g_stop_requested.load(std::memory_order_relaxed)) break;

    // Periodic save
    unsaved_iterations++;
    if (unsaved_iterations >= save_interval) {
      save_book(book, book_file);
      unsaved_iterations = 0;
    }
  }

  // Final save
  save_book(book, book_file);
  std::cout << "\nDone. Book has " << book.size() << " positions.\n";
  return 0;
}
