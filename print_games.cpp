// Print generated games as text for inspection.
// Based on generate_training.cpp but outputs human-readable game logs.

#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/notation.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"
#include "tablebase/compression.hpp"
#include "tablebase/tb_probe.hpp"
#include <atomic>
#include <chrono>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <bit>

// Check if position is quiet (no captures available)
bool is_quiet(const Board& board) {
  MoveList moves;
  generateMoves(board, moves);
  return moves.empty() || !moves[0].isCapture();
}

// Format a move as notation string
std::string move_notation(const Board& board, const Move& move, bool white_to_move) {
  std::vector<FullMove> full_moves;
  generateFullMoves(board, full_moves);
  for (const auto& fm : full_moves) {
    if (fm.move.from_xor_to == move.from_xor_to &&
        fm.move.captures == move.captures) {
      std::string result;
      for (size_t i = 0; i < fm.path.size(); ++i) {
        if (i > 0) result += move.isCapture() ? "x" : "-";
        int sq = white_to_move ? (fm.path[i] + 1) : (32 - fm.path[i]);
        result += std::to_string(sq);
      }
      return result;
    }
  }
  return "???";
}

// Play a single game and print it
void play_game(RandomBits& rng, search::Searcher& searcher,
               const CompressedTablebaseManager& wdl_tb,
               int game_num, int random_plies, std::uint64_t max_nodes) {
  Board board;
  int ply = 0;
  bool white_to_move = true;
  std::unordered_map<std::uint64_t, int> position_counts;
  position_counts[board.position_hash()] = 1;

  std::cout << "=== Game " << game_num << " ===\n";

  // Phase 1: Random opening
  while (ply < random_plies) {
    MoveList moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      std::cout << "  No moves - " << (white_to_move ? "White" : "Black") << " loses\n";
      std::cout << "  Result: " << (white_to_move ? "Black wins" : "White wins") << "\n\n";
      return;
    }

    std::uint64_t idx = rng() % moves.size();
    std::string notation = move_notation(board, moves[idx], white_to_move);
    int pieces = std::popcount(board.allPieces());

    std::cout << "  " << ply << ". " << (white_to_move ? "W" : "B")
              << " " << notation << " (random, " << pieces << "p)\n";

    board = makeMove(board, moves[idx]);
    ply++;
    white_to_move = !white_to_move;

    if (++position_counts[board.position_hash()] >= 3) {
      std::cout << "  Draw by repetition during random opening\n";
      std::cout << "  Result: Draw\n\n";
      return;
    }
  }

  std::cout << "  --- Search phase ---\n";

  // Phase 2: Play with search
  while (true) {
    MoveList moves;
    generateMoves(board, moves);

    if (moves.empty()) {
      std::cout << "  No moves - " << (white_to_move ? "White" : "Black") << " loses\n";
      std::cout << "  Result: " << (white_to_move ? "Black wins" : "White wins") << "\n\n";
      return;
    }

    int piece_count = std::popcount(board.allPieces());

    // Adjudicate at ≤7 pieces using WDL tablebases
    if (piece_count <= 7) {
      Value wdl = wdl_tb.lookup_wdl_preloaded(board);
      if (wdl != Value::UNKNOWN) {
        const char* result_str;
        if (wdl == Value::WIN) result_str = "WIN";
        else if (wdl == Value::LOSS) result_str = "LOSS";
        else result_str = "DRAW";

        std::cout << "  WDL adjudication at " << piece_count << " pieces: "
                  << result_str << " for "
                  << (white_to_move ? "White" : "Black") << "\n";

        if (wdl == Value::WIN)
          std::cout << "  Result: " << (white_to_move ? "White wins" : "Black wins") << "\n\n";
        else if (wdl == Value::LOSS)
          std::cout << "  Result: " << (white_to_move ? "Black wins" : "White wins") << "\n\n";
        else
          std::cout << "  Result: Draw\n\n";
        return;
      }
    }

    // Check for draw by 60 reversible moves
    if (board.n_reversible >= 60) {
      std::cout << "  Draw by 60-move rule\n";
      std::cout << "  Result: Draw\n\n";
      return;
    }

    // Search
    search::TimeControl tc;
    tc.soft_node_limit = max_nodes;
    tc.hard_node_limit = max_nodes * 5;
    auto result = searcher.search(board, 100, tc);

    if (result.best_move.from_xor_to == 0) {
      std::cout << "  Search returned no move\n";
      std::cout << "  Result: " << (white_to_move ? "Black wins" : "White wins") << "\n\n";
      return;
    }

    std::string notation = move_notation(board, result.best_move, white_to_move);
    bool quiet = is_quiet(board);
    Board next_board = makeMove(board, result.best_move);
    bool next_quiet = is_quiet(next_board);
    bool recorded = quiet && next_quiet;

    std::cout << "  " << ply << ". " << (white_to_move ? "W" : "B")
              << " " << notation
              << " (d=" << result.depth << " s=" << result.score
              << " " << piece_count << "p"
              << (recorded ? " REC" : "")
              << ")\n";

    board = next_board;
    ply++;
    white_to_move = !white_to_move;

    if (++position_counts[board.position_hash()] >= 3) {
      std::cout << "  Draw by threefold repetition\n";
      std::cout << "  Result: Draw\n\n";
      return;
    }

    if (ply > 500) {
      std::cout << "  Draw by ply limit (500)\n";
      std::cout << "  Result: Draw\n\n";
      return;
    }
  }
}

int main(int argc, char** argv) {
  std::string tb_dir = "/home/alvaro/claude/damas";
  std::string nn_model = "";
  std::uint64_t max_nodes = 10000;
  int random_plies = 10;
  int num_games = 20;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Print generated games as text\n\n"
                << "Options:\n"
                << "  --games N           Number of games (default: 20)\n"
                << "  --nodes N           Node limit per move (default: 10000)\n"
                << "  --random-plies N    Random opening moves (default: 10)\n"
                << "  --tb-path PATH      Tablebase directory\n"
                << "  --model FILE        Neural network model\n";
      return 0;
    } else if (arg == "--model" && i + 1 < argc) {
      nn_model = argv[++i];
    } else if (arg == "--nodes" && i + 1 < argc) {
      max_nodes = std::atoll(argv[++i]);
    } else if (arg == "--random-plies" && i + 1 < argc) {
      random_plies = std::atoi(argv[++i]);
    } else if (arg == "--tb-path" && i + 1 < argc) {
      tb_dir = argv[++i];
    } else if (arg == "--games" && i + 1 < argc) {
      num_games = std::atoi(argv[++i]);
    }
  }

  std::cout << "Tablebases: " << tb_dir << "\n";
  std::cout << "Model: " << (nn_model.empty() ? "random" : nn_model) << "\n";
  std::cout << "Nodes: " << max_nodes << "\n";
  std::cout << "Random plies: " << random_plies << "\n";
  std::cout << "Games: " << num_games << "\n\n";

  CompressedTablebaseManager wdl_tb(tb_dir);
  wdl_tb.preload(7);

  auto base_seed = static_cast<std::uint64_t>(
      std::chrono::high_resolution_clock::now().time_since_epoch().count());
  RandomBits rng(base_seed);

  search::Searcher searcher("", 0, nn_model);
  searcher.set_tt_size(32);

  for (int g = 1; g <= num_games; ++g) {
    play_game(rng, searcher, wdl_tb, g, random_plies, max_nodes);
  }

  return 0;
}
