// Opening book viewer
//
// Usage: viewbook [--book FILE] [move1 move2 ...]
// Example: viewbook 10-14 22-18 5-10

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
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
}

// Get move notation string
std::string move_notation(const Board& board, const Move& move, bool black_perspective) {
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
  std::ostringstream oss;
  oss << std::hex << move.from_xor_to;
  if (move.captures) oss << "x" << move.captures;
  return oss.str();
}

// Find a legal move matching the given notation string
Move parse_move(const Board& board, const std::string& notation, bool black_perspective) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    std::string name;
    if (!black_perspective) {
      name = moveToString(fm);
    } else {
      std::ostringstream oss;
      char sep = fm.move.isCapture() ? 'x' : '-';
      for (size_t i = 0; i < fm.path.size(); ++i) {
        if (i > 0) oss << sep;
        oss << (32 - fm.path[i]);
      }
      name = oss.str();
    }
    if (name == notation) return fm.move;
  }
  return Move();  // from_xor_to == 0 means not found
}

int main(int argc, char** argv) {
  std::string book_file = "opening.book";
  std::vector<std::string> move_strings;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--book" && i + 1 < argc) {
      book_file = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--book FILE] [move1 move2 ...]\n"
                << "View opening book positions\n\n"
                << "Examples:\n"
                << "  " << argv[0] << "                    # show root position\n"
                << "  " << argv[0] << " 10-14 22-18 5-10   # show position after these moves\n"
                << "  " << argv[0] << " --book my.book 10-14\n";
      return 0;
    } else {
      // Split on spaces (handles quoted "10-14 22-18 5-10")
      std::istringstream iss(arg);
      std::string token;
      while (iss >> token) {
        move_strings.push_back(token);
      }
    }
  }

  Book book;
  load_book(book, book_file);
  if (book.empty()) {
    std::cerr << "Book is empty or could not be loaded.\n";
    return 1;
  }

  // Walk through the move sequence
  Board current_board;  // initial position
  int ply = 0;

  for (const auto& move_str : move_strings) {
    bool black_perspective = (ply % 2 != 0);
    Move move = parse_move(current_board, move_str, black_perspective);
    if (move.from_xor_to == 0) {
      std::cerr << "Invalid move at ply " << ply << ": " << move_str << "\n";
      std::cerr << "Legal moves:";
      std::vector<FullMove> legal;
      generateFullMoves(current_board, legal);
      for (const auto& fm : legal) {
        std::cerr << " " << move_notation(current_board, fm.move, black_perspective);
      }
      std::cerr << "\n";
      return 1;
    }
    current_board = makeMove(current_board, move);
    ply++;
  }

  // Display position info
  bool wtm = (ply % 2 == 0);

  if (!move_strings.empty()) {
    std::cout << "Position after:";
    for (const auto& ms : move_strings) std::cout << " " << ms;
    std::cout << "\n";
  } else {
    std::cout << "Initial position\n";
  }
  std::cout << "Ply " << ply << ", " << (wtm ? "White" : "Black") << " to move\n\n";

  Board display = wtm ? current_board : flip(current_board);
  std::cout << display << "\n";

  // Look up in book
  uint64_t hash = current_board.position_hash();
  auto it = book.find(hash);

  if (it == book.end()) {
    std::cout << "Position NOT in book.\n";
    return 0;
  }

  const auto& entry = it->second;
  int total_visits = 0;
  for (const auto& mi : entry.moves) total_visits += mi.visits;

  // Sort moves by visits descending
  std::vector<size_t> indices(entry.moves.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    return entry.moves[a].visits > entry.moves[b].visits;
  });

  // Print moves table
  std::cout << std::left << std::setw(12) << "Move"
            << std::right << std::setw(8) << "Visits"
            << std::setw(10) << "Q"
            << std::setw(8) << "Prior"
            << std::setw(8) << "Share" << "\n";
  std::cout << std::string(46, '-') << "\n";

  for (size_t idx : indices) {
    const auto& mi = entry.moves[idx];
    std::string notation = move_notation(current_board, mi.move, !wtm);
    double q = mi.value_sum / mi.visits;
    double share = 100.0 * mi.visits / total_visits;

    std::cout << std::left << std::setw(12) << notation
              << std::right << std::setw(8) << mi.visits
              << std::fixed << std::setprecision(1) << std::setw(10) << q
              << std::setprecision(3) << std::setw(8) << mi.prior
              << std::setprecision(1) << std::setw(7) << share << "%\n";
  }

  std::cout << std::string(46, '-') << "\n";
  std::cout << std::left << std::setw(12) << "Total"
            << std::right << std::setw(8) << total_visits << "\n";

  // Show children
  std::cout << "\nChildren:\n";
  for (size_t idx : indices) {
    const auto& mi = entry.moves[idx];
    std::string notation = move_notation(current_board, mi.move, !wtm);

    Board child = makeMove(current_board, mi.move);
    uint64_t child_hash = child.position_hash();
    auto child_it = book.find(child_hash);

    if (child_it != book.end()) {
      const auto& ce = child_it->second;
      int child_total = 0;
      double best_q = -1e18;
      std::string best_move_name;
      for (const auto& cmi : ce.moves) {
        child_total += cmi.visits;
        double cq = cmi.value_sum / cmi.visits;
        if (cq > best_q) {
          best_q = cq;
          bool child_wtm = ((ply + 1) % 2 == 0);
          best_move_name = move_notation(child, cmi.move, !child_wtm);
        }
      }
      std::cout << "  " << std::left << std::setw(10) << notation
                << " → " << ce.moves.size() << " moves, best "
                << best_move_name << " Q=" << std::fixed << std::setprecision(1) << best_q
                << " (" << child_total << " visits)\n";
    } else {
      std::cout << "  " << std::left << std::setw(10) << notation << " → not in book\n";
    }
  }

  return 0;
}
