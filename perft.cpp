#include "core/board.hpp"
#include "core/movegen.hpp"
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
  int max_depth = 6;
  if (argc > 1) {
    max_depth = std::atoi(argv[1]);
  }

  Board board;  // Initial position
  std::cout << "Initial position:\n" << board << "\n";

  // Known perft values for Spanish checkers initial position
  // (from damas project test.cpp)
  std::uint64_t expected[] = {1, 7, 49, 302, 1469, 7361, 36473, 177532, 828783};

  for (int depth = 1; depth <= max_depth; ++depth) {
    auto start = std::chrono::high_resolution_clock::now();
    std::uint64_t nodes = perft(board, depth);
    auto end = std::chrono::high_resolution_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double nps = ms > 0 ? (nodes * 1000.0 / ms) : 0;

    bool ok = depth < 9 && nodes == expected[depth];
    std::cout << "perft(" << depth << ") = " << nodes;
    if (depth < 9) {
      std::cout << " (expected " << expected[depth] << ") " << (ok ? "OK" : "FAIL");
    }
    std::cout << "  [" << ms << " ms, " << static_cast<int>(nps) << " nps]\n";
  }

  return 0;
}
