#include "core/board.hpp"
#include "core/movegen.hpp"
#include "search/search.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <bit>

// Play a single game between two searchers
// Returns: +1 if white wins, -1 if black wins, 0 if draw
int play_game(search::Searcher& white, search::Searcher& black, int depth, bool verbose) {
    Board board;
    int ply = 0;

    while (true) {
        // Check for draw by reversible move rule
        if (board.n_reversible >= 60) {
            if (verbose) std::cout << "Draw by 60 reversible moves\n";
            return 0;
        }

        // Generate moves
        std::vector<Move> moves;
        generateMoves(board, moves);

        if (moves.empty()) {
            // Current side to move loses
            bool white_to_move = (ply % 2 == 0);
            if (verbose) {
                std::cout << (white_to_move ? "Black wins" : "White wins")
                          << " at ply " << ply << "\n";
            }
            return white_to_move ? -1 : +1;
        }

        // Select searcher based on side to move
        bool white_to_move = (ply % 2 == 0);
        search::Searcher& searcher = white_to_move ? white : black;

        auto result = searcher.search(board, depth);

        if (result.best_move.from_xor_to == 0) {
            // No move found (shouldn't happen)
            if (verbose) std::cout << "No move found at ply " << ply << "\n";
            return white_to_move ? -1 : +1;
        }

        board = makeMove(board, result.best_move);
        ply++;

        // Safety limit
        if (ply > 500) {
            if (verbose) std::cout << "Draw by move limit\n";
            return 0;
        }
    }
}

int main(int argc, char** argv) {
    std::string model1_path;
    std::string model2_path;
    std::string tb_dir = "/home/alvaro/claude/damas";
    int num_games = 100;
    int depth = 6;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <model1.bin> <model2.bin> [options]\n"
                      << "Play a match between two neural network models\n\n"
                      << "Arguments:\n"
                      << "  model1.bin          First model (plays white in odd games)\n"
                      << "  model2.bin          Second model (plays white in even games)\n\n"
                      << "Options:\n"
                      << "  -h, --help          Show this help message\n"
                      << "  --games N           Number of games (default: 100)\n"
                      << "  --depth N           Search depth (default: 6)\n"
                      << "  --tb-path PATH      Tablebase directory\n"
                      << "  --no-tb             Disable tablebases\n"
                      << "  --verbose           Print game results\n";
            return 0;
        } else if (arg == "--games" && i + 1 < argc) {
            num_games = std::atoi(argv[++i]);
        } else if (arg == "--depth" && i + 1 < argc) {
            depth = std::atoi(argv[++i]);
        } else if (arg == "--tb-path" && i + 1 < argc) {
            tb_dir = argv[++i];
        } else if (arg == "--no-tb") {
            tb_dir = "";
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (model1_path.empty()) {
            model1_path = arg;
        } else if (model2_path.empty()) {
            model2_path = arg;
        }
    }

    if (model1_path.empty() || model2_path.empty()) {
        std::cerr << "Error: Two model paths required\n";
        std::cerr << "Use -h for help\n";
        return 1;
    }

    std::cout << "=== Neural Network Match ===\n";
    std::cout << "Model 1: " << model1_path << "\n";
    std::cout << "Model 2: " << model2_path << "\n";
    std::cout << "Games: " << num_games << "\n";
    std::cout << "Depth: " << depth << "\n";
    std::cout << "Tablebases: " << (tb_dir.empty() ? "disabled" : tb_dir) << "\n\n";

    // Create searchers
    search::Searcher searcher1(tb_dir, 7, 6, model1_path);
    search::Searcher searcher2(tb_dir, 7, 6, model2_path);
    searcher1.set_tt_size(32);
    searcher2.set_tt_size(32);

    // Match statistics
    int model1_wins = 0, model2_wins = 0, draws = 0;
    int model1_white_wins = 0, model1_black_wins = 0;
    int model2_white_wins = 0, model2_black_wins = 0;

    for (int game = 0; game < num_games; ++game) {
        // Alternate colors: model1 is white in odd games (1, 3, 5...)
        bool model1_is_white = (game % 2 == 0);

        search::Searcher& white = model1_is_white ? searcher1 : searcher2;
        search::Searcher& black = model1_is_white ? searcher2 : searcher1;

        // Clear TT between games
        searcher1.clear_tt();
        searcher2.clear_tt();

        int result = play_game(white, black, depth, verbose);

        // Update statistics
        if (result == +1) {
            // White wins
            if (model1_is_white) {
                model1_wins++;
                model1_white_wins++;
            } else {
                model2_wins++;
                model2_white_wins++;
            }
        } else if (result == -1) {
            // Black wins
            if (model1_is_white) {
                model2_wins++;
                model2_black_wins++;
            } else {
                model1_wins++;
                model1_black_wins++;
            }
        } else {
            draws++;
        }

        // Progress
        if ((game + 1) % 10 == 0 || game == num_games - 1) {
            std::cout << "Game " << (game + 1) << "/" << num_games
                      << ": Model1 " << model1_wins << " - " << model2_wins << " Model2"
                      << " (" << draws << " draws)\n";
        }
    }

    // Final results
    std::cout << "\n=== Final Results ===\n";
    std::cout << "Model 1: " << model1_wins << " wins ("
              << model1_white_wins << " as white, " << model1_black_wins << " as black)\n";
    std::cout << "Model 2: " << model2_wins << " wins ("
              << model2_white_wins << " as white, " << model2_black_wins << " as black)\n";
    std::cout << "Draws: " << draws << "\n";

    double model1_score = model1_wins + 0.5 * draws;
    double model2_score = model2_wins + 0.5 * draws;
    double total = num_games;

    std::cout << "\nScore: " << std::fixed << std::setprecision(1)
              << model1_score << " - " << model2_score
              << " (" << (100.0 * model1_score / total) << "% - "
              << (100.0 * model2_score / total) << "%)\n";

    return 0;
}
