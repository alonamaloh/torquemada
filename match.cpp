#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tb_probe.hpp"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <bit>

// Generate a random opening position after N plies
Board generate_opening(RandomBits& rng, int random_plies) {
    Board board;
    for (int ply = 0; ply < random_plies; ++ply) {
        MoveList moves;
        generateMoves(board, moves);
        if (moves.empty()) break;
        std::uint64_t idx = rng() % moves.size();
        board = makeMove(board, moves[idx]);
    }
    return board;
}

// Play a single game between two searchers from a given position
// Returns: +1 if white wins, -1 if black wins, 0 if draw
int play_game(const Board& start, search::Searcher& white, search::Searcher& black,
              std::uint64_t max_nodes, bool verbose) {
    Board board = start;
    int ply = 0;

    while (true) {
        // Check for draw by reversible move rule
        if (board.n_reversible >= 60) {
            if (verbose) std::cout << "Draw by 60 reversible moves\n";
            return 0;
        }

        // Generate moves
        MoveList moves;
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
        searcher.set_root_white_to_move(white_to_move);

        auto result = searcher.search(board, 100, max_nodes);

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
    int num_pairs = 50;  // 50 pairs = 100 games
    std::uint64_t max_nodes = 10000;
    int random_plies = 10;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " <model1.bin> <model2.bin> [options]\n"
                      << "Play a match between two neural network models\n\n"
                      << "Arguments:\n"
                      << "  model1.bin          First model\n"
                      << "  model2.bin          Second model\n\n"
                      << "Options:\n"
                      << "  -h, --help          Show this help message\n"
                      << "  --pairs N           Number of game pairs (default: 50 = 100 games)\n"
                      << "  --nodes N           Node limit per move (default: 10000)\n"
                      << "  --random-plies N    Random opening moves (default: 10)\n"
                      << "  --tb-path PATH      Tablebase directory\n"
                      << "  --no-tb             Disable tablebases\n"
                      << "  --verbose           Print game results\n";
            return 0;
        } else if (arg == "--pairs" && i + 1 < argc) {
            num_pairs = std::atoi(argv[++i]);
        } else if (arg == "--random-plies" && i + 1 < argc) {
            random_plies = std::atoi(argv[++i]);
        } else if (arg == "--nodes" && i + 1 < argc) {
            max_nodes = std::atoll(argv[++i]);
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
    std::cout << "Game pairs: " << num_pairs << " (" << (num_pairs * 2) << " games)\n";
    std::cout << "Random opening plies: " << random_plies << "\n";
    std::cout << "Node limit: " << max_nodes << "\n";
    std::cout << "Tablebases: " << (tb_dir.empty() ? "disabled" : tb_dir) << "\n\n";

    // Initialize RNG
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
    RandomBits rng(seed);

    // Preload tablebases once (shared between both searchers)
    std::unique_ptr<CompressedTablebaseManager> tb_wdl;
    std::unique_ptr<tablebase::DTMTablebaseManager> tb_dtm;
    int tb_piece_limit = 7;
    int dtm_piece_limit = 6;

    if (!tb_dir.empty()) {
        std::cout << "Preloading tablebases...\n";
        tb_wdl = std::make_unique<CompressedTablebaseManager>(tb_dir);
        tb_wdl->preload(tb_piece_limit);
        tb_dtm = std::make_unique<tablebase::DTMTablebaseManager>(tb_dir);
        tb_dtm->preload(dtm_piece_limit);
    }

    // Create searchers with shared tablebases
    search::Searcher searcher1(tb_wdl.get(), tb_dtm.get(), tb_piece_limit, dtm_piece_limit, model1_path);
    search::Searcher searcher2(tb_wdl.get(), tb_dtm.get(), tb_piece_limit, dtm_piece_limit, model2_path);
    searcher1.set_tt_size(32);
    searcher2.set_tt_size(32);

    // Match statistics
    int model1_wins = 0, model2_wins = 0, draws = 0;
    int model1_white_wins = 0, model1_black_wins = 0;
    int model2_white_wins = 0, model2_black_wins = 0;
    int total_games = num_pairs * 2;

    for (int pair = 0; pair < num_pairs; ++pair) {
        // Generate random opening
        Board opening = generate_opening(rng, random_plies);

        // Play two games with same opening, swapping colors
        for (int swap = 0; swap < 2; ++swap) {
            bool model1_is_white = (swap == 0);
            int game_num = pair * 2 + swap + 1;

            search::Searcher& white = model1_is_white ? searcher1 : searcher2;
            search::Searcher& black = model1_is_white ? searcher2 : searcher1;

            // Clear TT between games
            searcher1.clear_tt();
            searcher2.clear_tt();

            int result = play_game(opening, white, black, max_nodes, verbose);

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

            // Progress after each game
            const char* result_str = (result == +1) ? "1-0" : (result == -1) ? "0-1" : "1/2";
            std::cout << "Game " << std::setw(3) << game_num << "/" << total_games
                      << "  " << (model1_is_white ? "M1" : "M2") << " vs "
                      << (model1_is_white ? "M2" : "M1") << "  " << result_str
                      << "  |  Score: " << model1_wins << "-" << model2_wins
                      << " =" << draws << "\n";
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
    double total = total_games;

    std::cout << "\nScore: " << std::fixed << std::setprecision(1)
              << model1_score << " - " << model2_score
              << " (" << (100.0 * model1_score / total) << "% - "
              << (100.0 * model2_score / total) << "%)\n";

    return 0;
}
