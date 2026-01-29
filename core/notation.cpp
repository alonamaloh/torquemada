#include "notation.h"
#include <bit>
#include <cctype>
#include <sstream>

std::string moveToString(const FullMove& fullMove) {
  const auto& path = fullMove.path;
  if (path.empty()) return "";

  std::ostringstream oss;
  char sep = fullMove.move.isCapture() ? 'x' : '-';

  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0) oss << sep;
    oss << (path[i] + 1);  // Convert 0-31 to 1-32
  }
  return oss.str();
}

std::string moveToString(const Move& move) {
  Bb fxt = move.from_xor_to;

  // Handle circular captures (from == to, so from_xor_to == 0)
  if (fxt == 0) {
    // Can't determine the path from the move alone
    return "circular";
  }

  int sq1 = std::countr_zero(fxt);
  fxt &= fxt - 1;
  int sq2 = std::countr_zero(fxt);
  if (sq2 == 32) sq2 = sq1;

  // Ensure from < to for consistent display
  int from = std::min(sq1, sq2);
  int to = std::max(sq1, sq2);

  std::ostringstream oss;
  oss << (from + 1);
  oss << (move.isCapture() ? 'x' : '-');
  oss << (to + 1);
  return oss.str();
}

// Helper: parse a square number (1-32) from string, return 0-31 or -1 on error
static int parseSquare(const std::string& s, size_t& pos) {
  if (pos >= s.size() || !std::isdigit(s[pos]))
    return -1;

  int num = 0;
  while (pos < s.size() && std::isdigit(s[pos])) {
    num = num * 10 + (s[pos] - '0');
    ++pos;
  }

  if (num < 1 || num > 32)
    return -1;

  return num - 1;  // Convert to 0-31
}

// Flip a square index (0-31) for the 180-degree rotation
static int flipSquare(int sq) {
  return 31 - sq;
}

std::optional<FullMove> parseMove(const Board& board, const std::string& str,
                                   bool fromBlackPerspective) {
  // Parse the move string to extract squares
  std::vector<int> squares;
  bool isCapture = false;

  size_t pos = 0;

  // Skip leading whitespace
  while (pos < str.size() && std::isspace(str[pos])) ++pos;

  // Parse first square
  int sq = parseSquare(str, pos);
  if (sq < 0) return std::nullopt;
  squares.push_back(sq);

  // Parse separator and subsequent squares
  while (pos < str.size()) {
    char sep = str[pos];
    if (sep == '-') {
      ++pos;
    } else if (sep == 'x' || sep == 'X' || sep == ':') {
      isCapture = true;
      ++pos;
    } else {
      break;  // End of move notation
    }

    sq = parseSquare(str, pos);
    if (sq < 0) return std::nullopt;
    squares.push_back(sq);
  }

  if (squares.size() < 2)
    return std::nullopt;

  // If parsing from black's perspective, flip squares to match internal representation
  if (fromBlackPerspective) {
    for (int& s : squares)
      s = flipSquare(s);
  }

  // Generate legal moves and find a match
  std::vector<FullMove> moves;
  generateFullMoves(board, moves);

  int from = squares.front();
  int to = squares.back();
  Bb fromBit = 1u << from;
  Bb toBit = 1u << to;
  Bb expectedFxt = fromBit ^ toBit;

  for (const auto& fullMove : moves) {
    if (fullMove.move.from_xor_to != expectedFxt)
      continue;

    // Check capture expectation matches
    if (isCapture != fullMove.move.isCapture())
      continue;

    // For multi-square notation, verify the path matches
    if (squares.size() > 2 && fullMove.move.isCapture()) {
      if (squares.size() != fullMove.path.size())
        continue;
      // Check that the path matches
      bool pathMatches = true;
      for (size_t i = 0; i < squares.size(); ++i) {
        if (squares[i] != fullMove.path[i]) {
          pathMatches = false;
          break;
        }
      }
      if (!pathMatches)
        continue;
    }

    return fullMove;
  }

  return std::nullopt;
}

std::optional<FullMove> parseMove(const Board& board, const std::string& str) {
  return parseMove(board, str, false);
}

GameRecord parseGame(const Board& startBoard, const std::string& record) {
  GameRecord result;
  result.finalBoard = startBoard;
  result.complete = true;

  std::istringstream iss(record);
  std::string token;
  int ply = 0;  // 0 = white's move, 1 = black's move, etc.

  while (iss >> token) {
    // Skip move numbers like "1." or "12."
    if (!token.empty() && token.back() == '.') {
      bool allDigits = true;
      for (size_t i = 0; i < token.size() - 1; ++i) {
        if (!std::isdigit(token[i])) {
          allDigits = false;
          break;
        }
      }
      if (allDigits) continue;
    }

    // Skip common annotations
    if (token == "..." || token == "*" || token == "1-0" ||
        token == "0-1" || token == "1/2-1/2")
      continue;

    // Try to parse as a move
    // After each move the board is flipped, so odd plies need flipped notation
    bool fromBlackPerspective = (ply % 2 == 1);
    auto fullMove = parseMove(result.finalBoard, token, fromBlackPerspective);
    if (!fullMove) {
      result.complete = false;
      result.error = "Invalid move: " + token;
      break;
    }

    result.moves.push_back(*fullMove);
    result.finalBoard = makeMove(result.finalBoard, fullMove->move);
    ++ply;
  }

  return result;
}

// Convert FullMove to string from a specific perspective
static std::string moveToStringPerspective(const FullMove& fullMove, bool fromBlackPerspective) {
  const auto& path = fullMove.path;
  if (path.empty()) return "";

  std::ostringstream oss;
  char sep = fullMove.move.isCapture() ? 'x' : '-';

  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0) oss << sep;
    int sq = path[i];
    // If from black's perspective, flip squares back to original orientation
    if (fromBlackPerspective) {
      sq = flipSquare(sq);
    }
    oss << (sq + 1);  // Convert 0-31 to 1-32
  }
  return oss.str();
}

std::string gameToString(const std::vector<FullMove>& moves) {
  std::ostringstream oss;
  for (size_t i = 0; i < moves.size(); ++i) {
    if (i > 0) oss << ' ';
    if (i % 2 == 0) {
      oss << (i / 2 + 1) << ". ";
    }
    // Odd-indexed moves (black's moves) need flipped perspective
    bool fromBlackPerspective = (i % 2 == 1);
    oss << moveToStringPerspective(moves[i], fromBlackPerspective);
  }
  return oss.str();
}
