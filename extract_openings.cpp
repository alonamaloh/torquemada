#include "core/board.hpp"
#include "core/notation.hpp"
#include <fstream>
#include <iostream>
#include <regex>
#include <string>

int main(int argc, char* argv[]) {
  const char* input = argc > 1 ? argv[1] : "opening_names";

  std::ifstream fin(input);
  if (!fin) {
    std::cerr << "Cannot open " << input << "\n";
    return 1;
  }

  std::string line;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;

    // Extract the quoted name
    auto qstart = line.find('"');
    auto qend = line.rfind('"');
    if (qstart == std::string::npos || qstart == qend) {
      std::cerr << "No quoted name in: " << line << "\n";
      continue;
    }
    std::string name = line.substr(qstart + 1, qend - qstart - 1);
    std::string moves = line.substr(0, qstart);

    // Parse and replay the moves
    GameRecord record = parseGame(Board(), moves);
    if (!record.complete) {
      std::cerr << "Parse error in \"" << name << "\": " << record.error << "\n";
      continue;
    }

    const Board& b = record.finalBoard;
    int ply = record.moves.size();
    bool whiteToMove = (ply % 2 == 0);
    // Output from UI perspective (actual white/black pieces)
    Bb w, bl, k;
    if (whiteToMove) {
      w = b.white; bl = b.black; k = b.kings;
    } else {
      w = flip(b.black); bl = flip(b.white); k = flip(b.kings);
    }
    printf("%08x %08x %08x %s\n", w, bl, k, name.c_str());
  }

  return 0;
}
