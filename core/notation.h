#pragma once

#include "board.h"
#include "movegen.h"
#include <optional>
#include <string>
#include <vector>

// Convert a FullMove to standard notation (1-32 based)
// Simple moves: "9-13"
// Captures: "9x14x18" (shows full path of landing squares)
std::string moveToString(const FullMove& move);

// Convert a compact Move to notation (limited - can't show full path for captures)
// For circular captures (from == to), returns "circular"
std::string moveToString(const Move& move);

// Parse a move string and find the matching legal move.
// Accepts formats like "9-13", "9x18", "9x14x23" (path notation).
// If fromBlackPerspective is true, squares are flipped to match internal representation.
// Returns nullopt if no matching legal move is found.
std::optional<FullMove> parseMove(const Board& board, const std::string& str,
                                   bool fromBlackPerspective);
std::optional<FullMove> parseMove(const Board& board, const std::string& str);

// Parse a sequence of moves (game record) from a string.
// Moves can be separated by spaces, newlines, or move numbers (e.g., "1. 9-13 9-14").
// Returns the resulting board and list of moves played.
// Stops parsing on first invalid move.
struct GameRecord {
  Board finalBoard;
  std::vector<FullMove> moves;
  bool complete;  // true if all moves parsed successfully
  std::string error;  // error message if not complete
};

GameRecord parseGame(const Board& startBoard, const std::string& record);

// Convert a game record to string notation (with full capture paths)
std::string gameToString(const std::vector<FullMove>& moves);

