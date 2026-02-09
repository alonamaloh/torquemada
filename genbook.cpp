// PUCT-based opening book generator (multi-threaded)
//
// Uses a PUCT (Predictor + Upper Confidence bound for Trees) algorithm inspired
// by AlphaZero's MCTS. Each iteration walks from the root, selecting moves via
// PUCT, expands a leaf node, and backpropagates the evaluation.
//
// Parallelism: multiple threads walk the shared tree simultaneously, using
// virtual losses to encourage divergence into different lines.
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
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
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
//
// Visit share cap: if a move has accumulated more than max_visit_share of
// total visits AND there is at least one alternative within q_threshold of
// its Q value, skip it.  This prevents PUCT from converging on a single
// line while still allowing dominance when one move is clearly superior.
int select_puct(const BookEntry& entry, double c_puct,
                double max_visit_share, double q_threshold) {
  int total_visits = 0;
  for (const auto& mi : entry.moves) {
    total_visits += mi.visits;
  }
  double sqrt_total = std::sqrt(static_cast<double>(total_visits));
  int visit_cap = std::max(1, static_cast<int>(total_visits * max_visit_share));

  int best_idx = 0;
  double best_score = -1e18;
  for (size_t i = 0; i < entry.moves.size(); ++i) {
    const auto& mi = entry.moves[i];
    double q = mi.value_sum / mi.visits;
    double u = c_puct * mi.prior * sqrt_total / (1 + mi.visits);

    // Visit share cap: skip if over cap and a competitive alternative exists
    if (mi.visits > visit_cap) {
      bool has_competitive = false;
      for (size_t j = 0; j < entry.moves.size(); ++j) {
        if (j == i) continue;
        double q_j = entry.moves[j].value_sum / entry.moves[j].visits;
        if (q - q_j <= q_threshold) {
          has_competitive = true;
          break;
        }
      }
      if (has_competitive) continue;
    }

    double score = q + u;
    if (score > best_score) {
      best_score = score;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

// Virtual loss: large enough to divert threads to different good moves,
// small enough that garbage moves still look like garbage.
// Should be ~2-3x the typical Q spread between reasonable moves.
static constexpr double VIRTUAL_LOSS = 3000.0;

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string nn_model;
  std::string book_file = "opening.book";
  uint64_t nodes = 10000000;
  double c_puct = 1000.0;
  double prior_temp = 1000.0;
  int max_ply = 30;
  int max_iterations = 0;  // 0 = unlimited
  int save_interval = 10;
  int tb_limit = 7;
  int num_threads = 1;
  double max_visit_share = 0.7;
  double q_threshold = 500.0;
  double max_leaf_value = 10000.0;
  int draw_score = 0;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
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
    } else if (arg == "--threads" && i + 1 < argc) {
      num_threads = std::stoi(argv[++i]);
    } else if (arg == "--max-visit-share" && i + 1 < argc) {
      max_visit_share = std::stod(argv[++i]);
    } else if (arg == "--q-threshold" && i + 1 < argc) {
      q_threshold = std::stod(argv[++i]);
    } else if (arg == "--max-leaf-value" && i + 1 < argc) {
      max_leaf_value = std::stod(argv[++i]);
    } else if (arg == "--draw-score" && i + 1 < argc) {
      draw_score = std::stoi(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Generate an opening book using PUCT-based exploration\n\n"
                << "Options:\n"
                << "  --model FILE       Neural network model\n"
                << "  --tb PATH          Tablebase directory (default: /home/alvaro/claude/damas)\n"
                << "  --no-tb            Disable tablebases\n"
                << "  --nodes N          Node limit for search (default: 10000000)\n"
                << "  --cpuct N          PUCT exploration constant (default: 1000)\n"
                << "  --prior-temp T     Softmax temperature for priors (default: 1000)\n"
                << "  --max-ply N        Maximum book depth in plies (default: 30)\n"
                << "  --iterations N     Number of iterations (default: unlimited)\n"
                << "  --save-interval N  Save every N iterations (default: 10)\n"
                << "  --book FILE        Book file to load/save (default: opening.book)\n"
                << "  --threads N        Number of worker threads (default: 1)\n"
                << "  --max-visit-share F  Cap move visits at F*total when alternatives\n"
                << "                       are within Q threshold (default: 0.7)\n"
                << "  --q-threshold T    Q difference to consider a move competitive (default: 500)\n"
                << "  --max-leaf-value V Cap leaf search scores to +/-V (default: 10000)\n"
                << "  --draw-score N     Score assigned to draws (default: 0)\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  bool verbose = (num_threads == 1);

  // Initialize
  std::cout << "=== PUCT Opening Book Generator ===\n";
  std::cout << "Threads: " << num_threads << "\n";
  std::cout << "Nodes: " << nodes << "\n";
  std::cout << "C_PUCT: " << c_puct << "\n";
  std::cout << "Prior temperature: " << prior_temp << "\n";
  std::cout << "Max ply: " << max_ply << "\n";
  std::cout << "Max visit share: " << max_visit_share << "\n";
  std::cout << "Q threshold: " << q_threshold << "\n";
  std::cout << "Max leaf value: " << max_leaf_value << "\n";
  std::cout << "Draw score: " << draw_score << "\n";
  std::cout << "Save interval: " << save_interval << "\n";
  std::cout << "Book file: " << book_file << "\n";

  if (!nn_model.empty()) std::cout << "Model: " << nn_model << "\n";
  if (!tb_dir.empty()) std::cout << "Tablebases: " << tb_dir << "\n";

  // Load tablebases once, shared across all threads (read-only after preload)
  std::unique_ptr<tablebase::DTMTablebaseManager> dtm_manager;
  if (!tb_dir.empty() && tb_limit > 0) {
    dtm_manager = std::make_unique<tablebase::DTMTablebaseManager>(tb_dir);
    dtm_manager->preload(tb_limit);
  }

  // Create one searcher per thread, sharing the tablebase manager
  std::vector<std::unique_ptr<search::Searcher>> searchers;
  for (int i = 0; i < num_threads; ++i) {
    auto s = std::make_unique<search::Searcher>(
        dtm_manager.get(), tb_limit, nn_model);
    s->set_tt_size(64);
    s->set_verbose(false);
    s->set_draw_score(draw_score);
    s->set_stop_flag(&g_stop_requested);
    searchers.push_back(std::move(s));
  }

  std::cout << "Memory: " << num_threads << " x 64 MB TT = "
            << num_threads * 64 << " MB\n";

  std::signal(SIGINT, sigint_handler);

  // Load existing book
  Book book;
  load_book(book, book_file);

  std::mutex book_mutex;
  std::atomic<int> total_iterations{0};

  // Worker function: each thread runs this loop
  auto worker = [&](int thread_id) {
    auto& searcher = *searchers[thread_id];

    while (!g_stop_requested.load(std::memory_order_relaxed)) {
      int iter = total_iterations.fetch_add(1) + 1;
      if (max_iterations > 0 && iter > max_iterations) break;

      // --- Phase 1: Walk tree under lock, apply virtual losses ---
      std::vector<std::pair<uint64_t, int>> path;
      std::string path_desc;  // compact path for multi-threaded output

      Board target_board;
      int target_ply = 0;
      bool need_search = false;
      bool add_to_book = false;  // true for expansion, false for max-ply eval
      bool got_leaf = false;
      double leaf_value = 0;

      {
        std::lock_guard<std::mutex> lock(book_mutex);

        if (verbose) {
          std::cout << "\n--- Iteration " << iter << " ---\n";
        }

        Board current_board;  // initial position
        int ply = 0;

        while (ply < max_ply) {
          if (g_stop_requested.load(std::memory_order_relaxed)) break;

          MoveList legal_moves;
          generateMoves(current_board, legal_moves);
          if (legal_moves.empty()) {
            leaf_value = -100000;
            got_leaf = true;
            if (verbose) std::cout << "  Game over at ply " << ply << "\n";
            break;
          }

          uint64_t hash = current_board.position_hash();
          auto it = book.find(hash);

          if (it != book.end()) {
            // Position is in book — select move by PUCT, apply virtual loss
            int idx = select_puct(it->second, c_puct, max_visit_share, q_threshold);
            auto& mi = it->second.moves[idx];
            bool wtm = (ply % 2 == 0);
            std::string notation = move_notation(current_board, mi.move, !wtm);

            if (verbose) {
              double q = mi.value_sum / mi.visits;
              std::cout << "  Ply " << ply << " (" << (wtm ? "W" : "B")
                        << "): PUCT " << notation
                        << " (Q=" << std::fixed << std::setprecision(1) << q
                        << " prior=" << std::setprecision(3) << mi.prior
                        << " visits=" << mi.visits << ")\n";
            } else {
              if (!path_desc.empty()) path_desc += " ";
              path_desc += notation;
            }

            // Apply virtual loss
            mi.visits++;
            mi.value_sum -= VIRTUAL_LOSS;

            path.push_back({hash, idx});
            current_board = makeMove(current_board, mi.move);
            ply++;
            continue;
          }

          // Position not in book
          bool wtm = (ply % 2 == 0);

          // Forced move: add to book and keep walking
          if (legal_moves.size() == 1) {
            std::string notation = move_notation(current_board, legal_moves[0], !wtm);

            if (verbose) {
              std::cout << "  Ply " << ply << " (" << (wtm ? "W" : "B")
                        << "): forced " << notation << "\n";
            } else {
              if (!path_desc.empty()) path_desc += " ";
              path_desc += notation;
            }

            BookEntry entry;
            entry.board = current_board;
            entry.moves.push_back({legal_moves[0], 1, 0.0, 1.0});
            book[hash] = std::move(entry);

            // Apply virtual loss to the new entry
            auto& mi = book[hash].moves[0];
            mi.visits++;
            mi.value_sum -= VIRTUAL_LOSS;

            path.push_back({hash, 0});
            current_board = makeMove(current_board, legal_moves[0]);
            ply++;
            continue;
          }

          // Multi-move position not in book — need expansion search
          target_board = current_board;
          target_ply = ply;
          need_search = true;
          add_to_book = true;

          if (verbose) {
            std::cout << "  Expanding at ply " << ply << " ("
                      << (wtm ? "White" : "Black") << " to move), "
                      << legal_moves.size() << " legal moves\n";
            Board display = wtm ? current_board : flip(current_board);
            std::cout << display;
          }
          break;
        }

        // Handle max ply reached
        if (!got_leaf && !need_search && ply >= max_ply &&
            !g_stop_requested.load(std::memory_order_relaxed)) {
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
            if (verbose) {
              std::cout << "  Reached max ply (in book), leaf Q="
                        << std::fixed << std::setprecision(1) << leaf_value << "\n";
            }
          } else {
            // Need to search for evaluation (but don't add to book)
            MoveList legal_moves;
            generateMoves(current_board, legal_moves);
            if (legal_moves.empty()) {
              leaf_value = -100000;
              got_leaf = true;
            } else {
              target_board = current_board;
              target_ply = ply;
              need_search = true;
              add_to_book = false;
            }
          }
        }
      } // unlock book_mutex

      // --- Phase 2: Search (no lock — this is where time is spent) ---
      // Store search results in local variables for use in Phase 3
      int best_score = 0;
      int search_depth = 0;
      uint64_t search_nodes = 0;
      std::vector<std::pair<Move, int>> scored_moves;  // (move, score)

      if (need_search && !g_stop_requested.load(std::memory_order_relaxed)) {
        bool wtm = (target_ply % 2 == 0);
        searcher.set_root_white_to_move(wtm);
        searcher.set_perspective(wtm);
        searcher.clear_tt();

        auto result = searcher.search_multi(target_board, 100,
                                           search::TimeControl::with_nodes(nodes), 30000);

        if (!result.moves.empty() &&
            !g_stop_requested.load(std::memory_order_relaxed)) {
          best_score = result.moves[0].score;
          search_depth = result.depth;
          search_nodes = result.nodes;
          leaf_value = best_score;
          got_leaf = true;
          for (const auto& ms : result.moves) {
            scored_moves.push_back({ms.move, ms.score});
          }
        }
      }

      // --- Phase 3: Update book and backprop under lock ---
      {
        std::lock_guard<std::mutex> lock(book_mutex);

        // Add expansion to book
        if (add_to_book && got_leaf && !scored_moves.empty()) {
          bool wtm = (target_ply % 2 == 0);
          uint64_t hash = target_board.position_hash();

          // Only add if another thread didn't already expand this position
          if (book.find(hash) == book.end()) {
            // Compute softmax priors
            double prior_sum = 0;
            std::vector<double> priors(scored_moves.size());
            for (size_t i = 0; i < scored_moves.size(); ++i) {
              priors[i] = std::exp((scored_moves[i].second - best_score) / prior_temp);
              prior_sum += priors[i];
            }
            for (auto& p : priors) p /= prior_sum;

            BookEntry entry;
            entry.board = target_board;
            for (size_t i = 0; i < scored_moves.size(); ++i) {
              double clamped_score = std::clamp(
                  static_cast<double>(scored_moves[i].second),
                  -max_leaf_value, max_leaf_value);
              entry.moves.push_back({
                scored_moves[i].first,
                1,                    // virtual visit
                clamped_score,        // initial value_sum
                priors[i]
              });
            }
            book[hash] = std::move(entry);
          }

          // Print expansion info
          if (verbose) {
            for (const auto& [move, score] : scored_moves) {
              std::string notation = move_notation(target_board, move, !wtm);
              std::cout << "    " << notation << ": " << score << "\n";
            }
            std::cout << "  depth " << search_depth << ", " << search_nodes << " nodes\n";

            auto& be = book[hash];
            std::cout << "  " << be.moves.size() << " moves (best score: " << best_score << "):\n";
            for (const auto& mi : be.moves) {
              std::string notation = move_notation(target_board, mi.move, !wtm);
              std::cout << "    " << notation
                        << " prior=" << std::fixed << std::setprecision(3) << mi.prior
                        << " Q=" << std::setprecision(1) << (mi.value_sum / mi.visits) << "\n";
            }
          } else {
            std::cout << "[T" << thread_id << "] Iter " << iter << ": "
                      << path_desc << (path_desc.empty() ? "" : " > ")
                      << "ply " << target_ply
                      << " (" << (wtm ? "W" : "B") << "), "
                      << scored_moves.size() << " moves, best=" << best_score
                      << ", d" << search_depth
                      << " [" << book.size() << " pos]\n";
          }
        }

        // Undo virtual losses on the path
        for (const auto& [hash, idx] : path) {
          auto& mi = book[hash].moves[idx];
          mi.visits--;
          mi.value_sum += VIRTUAL_LOSS;
        }

        // Backpropagate real values
        if (got_leaf && !path.empty()) {
          double value = std::clamp(leaf_value, -max_leaf_value, max_leaf_value);
          if (verbose) {
            std::cout << "  Backpropagating value=" << std::fixed
                      << std::setprecision(1) << value
                      << " through " << path.size() << " plies\n";
          }
          for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
            value = -value;
            auto& mi = book[path[i].first].moves[path[i].second];
            mi.value_sum += value;
            mi.visits++;
          }
        }

        // Periodic save
        if (iter % save_interval == 0) {
          save_book(book, book_file);
        }
      } // unlock book_mutex
    }
  };

  // Launch worker threads
  auto start_time = std::chrono::steady_clock::now();

  std::vector<std::thread> threads;
  for (int i = 1; i < num_threads; ++i) {
    threads.emplace_back(worker, i);
  }
  worker(0);  // main thread also works

  for (auto& t : threads) {
    t.join();
  }

  auto elapsed = std::chrono::steady_clock::now() - start_time;
  double seconds = std::chrono::duration<double>(elapsed).count();

  // Final save
  save_book(book, book_file);
  int iters = total_iterations.load();
  std::cout << "\nDone. " << iters << " iterations in "
            << std::fixed << std::setprecision(1) << seconds << "s ("
            << std::setprecision(1) << (iters / std::max(seconds, 0.001)) << " iter/s). "
            << "Book has " << book.size() << " positions.\n";
  return 0;
}
