// Condense PUCT opening books into a compact play-ready format
//
// Takes white and black PUCT book files, filters positions by visit count,
// removes low-visit moves, normalizes to probabilities, and outputs a
// combined condensed book suitable for the web interface.
//
// Usage: condensebook --white white.book --black black.book -o opening.cbook

#include "core/board.hpp"
#include "core/movegen.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct MoveInfo {
  Move move;
  int visits;
  double value_sum;
  double prior;
};

struct BookEntry {
  Board board;
  std::vector<MoveInfo> moves;
};

using Book = std::unordered_map<uint64_t, BookEntry>;

void load_book(Book& book, const std::string& filename) {
  std::ifstream in(filename);
  if (!in) {
    std::cerr << "Error: cannot open " << filename << "\n";
    return;
  }

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

  std::cout << "Loaded " << filename << ": " << book.size() << " positions\n";
}

// BFS from initial position to determine ply parity of each position
std::unordered_map<uint64_t, int> compute_ply_map(const Book& book) {
  std::unordered_map<uint64_t, int> ply_map;
  std::queue<std::pair<Board, int>> frontier;

  Board initial;
  uint64_t init_hash = initial.position_hash();
  ply_map[init_hash] = 0;
  frontier.push({initial, 0});

  while (!frontier.empty()) {
    auto [board, ply] = frontier.front();
    frontier.pop();

    uint64_t hash = board.position_hash();
    auto it = book.find(hash);
    if (it == book.end()) continue;

    for (const auto& mi : it->second.moves) {
      Board child = makeMove(board, mi.move);
      uint64_t child_hash = child.position_hash();
      if (ply_map.find(child_hash) == ply_map.end()) {
        ply_map[child_hash] = ply + 1;
        frontier.push({child, ply + 1});
      }
    }
  }

  return ply_map;
}

struct CondensedMove {
  Move move;
  double probability;
};

// Filter and normalize a book entry into play probabilities
// Returns empty vector if position doesn't meet criteria
std::vector<CondensedMove> condense_entry(const BookEntry& entry,
                                          int min_visits, double min_ratio) {
  int max_visits = 0;
  for (const auto& mi : entry.moves) {
    max_visits = std::max(max_visits, mi.visits);
  }

  if (max_visits < min_visits) return {};

  int threshold = static_cast<int>(max_visits * min_ratio);

  std::vector<CondensedMove> result;
  int filtered_total = 0;
  for (const auto& mi : entry.moves) {
    if (mi.visits >= threshold) {
      result.push_back({mi.move, static_cast<double>(mi.visits)});
      filtered_total += mi.visits;
    }
  }

  if (filtered_total > 0) {
    for (auto& cm : result) {
      cm.probability /= filtered_total;
    }
  }

  // Sort by probability descending
  std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.probability > b.probability;
  });

  return result;
}

int main(int argc, char** argv) {
  std::string white_file, black_file, output_file = "opening.cbook";
  int min_visits = 10;
  double min_ratio = 0.1;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "--white" || arg == "-w") && i + 1 < argc) {
      white_file = argv[++i];
    } else if ((arg == "--black" || arg == "-b") && i + 1 < argc) {
      black_file = argv[++i];
    } else if ((arg == "--output" || arg == "-o") && i + 1 < argc) {
      output_file = argv[++i];
    } else if (arg == "--min-visits" && i + 1 < argc) {
      min_visits = std::stoi(argv[++i]);
    } else if (arg == "--min-ratio" && i + 1 < argc) {
      min_ratio = std::stod(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Condense PUCT opening books into a compact play-ready format\n\n"
                << "Options:\n"
                << "  --white FILE, -w   White PUCT book file\n"
                << "  --black FILE, -b   Black PUCT book file\n"
                << "  --output FILE, -o  Output condensed book (default: opening.cbook)\n"
                << "  --min-visits N     Minimum visits to include position (default: 10)\n"
                << "  --min-ratio R      Exclude moves below R * max_visits (default: 0.1)\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  if (white_file.empty() && black_file.empty()) {
    std::cerr << "Error: provide at least one of --white or --black\n";
    return 1;
  }

  std::cout << "Min visits: " << min_visits << "\n";
  std::cout << "Min ratio: " << min_ratio << "\n";

  // Load books
  Book white_book, black_book;
  if (!white_file.empty()) load_book(white_book, white_file);
  if (!black_file.empty()) load_book(black_book, black_file);

  // Compute ply parities via BFS from initial position
  auto white_ply = white_book.empty()
      ? std::unordered_map<uint64_t, int>{}
      : compute_ply_map(white_book);
  auto black_ply = black_book.empty()
      ? std::unordered_map<uint64_t, int>{}
      : compute_ply_map(black_book);

  std::cout << "White book: " << white_ply.size() << " reachable positions\n";
  std::cout << "Black book: " << black_ply.size() << " reachable positions\n";

  // Write condensed book
  std::ofstream out(output_file);
  if (!out) {
    std::cerr << "Error: cannot write to " << output_file << "\n";
    return 1;
  }

  out << "# Condensed opening book\n";
  out << "# P <white_hex> <black_hex> <kings_hex>\n";
  out << "# M <from_xor_to_hex> <captures_hex> <probability>\n";
  out << std::fixed << std::setprecision(3);

  int white_positions = 0;
  int black_positions = 0;
  int total_moves = 0;
  std::unordered_set<uint64_t> written;

  // White book: even ply (white to move)
  for (const auto& [hash, ply] : white_ply) {
    if (ply % 2 != 0) continue;

    auto it = white_book.find(hash);
    if (it == white_book.end()) continue;

    auto moves = condense_entry(it->second, min_visits, min_ratio);
    if (moves.empty()) continue;

    const auto& board = it->second.board;
    out << "\nP " << std::hex
        << board.white << " " << board.black << " " << board.kings
        << std::dec << "\n";
    for (const auto& cm : moves) {
      out << "M " << std::hex
          << cm.move.from_xor_to << " " << cm.move.captures
          << std::dec << " " << cm.probability << "\n";
    }

    written.insert(hash);
    white_positions++;
    total_moves += static_cast<int>(moves.size());
  }

  // Black book: odd ply (black to move)
  for (const auto& [hash, ply] : black_ply) {
    if (ply % 2 != 1) continue;
    if (written.count(hash)) continue;  // already written

    auto it = black_book.find(hash);
    if (it == black_book.end()) continue;

    auto moves = condense_entry(it->second, min_visits, min_ratio);
    if (moves.empty()) continue;

    const auto& board = it->second.board;
    out << "\nP " << std::hex
        << board.white << " " << board.black << " " << board.kings
        << std::dec << "\n";
    for (const auto& cm : moves) {
      out << "M " << std::hex
          << cm.move.from_xor_to << " " << cm.move.captures
          << std::dec << " " << cm.probability << "\n";
    }

    written.insert(hash);
    black_positions++;
    total_moves += static_cast<int>(moves.size());
  }

  out.close();

  std::cout << "\nCondensed book written to " << output_file << ":\n";
  std::cout << "  White positions: " << white_positions << "\n";
  std::cout << "  Black positions: " << black_positions << "\n";
  std::cout << "  Total positions: " << (white_positions + black_positions) << "\n";
  std::cout << "  Total moves: " << total_moves << "\n";

  return 0;
}
