#include "core/board.hpp"
#include "core/movegen.hpp"
#include "core/random.hpp"
#include "search/search.hpp"
#include "tablebase/tablebase.hpp"
#include <omp.h>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
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
              std::uint64_t max_nodes) {
    Board board = start;
    int ply = 0;

    while (true) {
        if (board.n_reversible >= 60) return 0;

        MoveList moves;
        generateMoves(board, moves);

        if (moves.empty()) {
            bool white_to_move = (ply % 2 == 0);
            return white_to_move ? -1 : +1;
        }

        bool white_to_move = (ply % 2 == 0);
        search::Searcher& searcher = white_to_move ? white : black;
        searcher.set_root_white_to_move(white_to_move);

        search::TimeControl tc;
        tc.soft_node_limit = max_nodes;
        tc.hard_node_limit = max_nodes * 5;
        auto result = searcher.search(board, 100, tc);

        if (result.best_move.from_xor_to == 0) {
            return white_to_move ? -1 : +1;
        }

        board = makeMove(board, result.best_move);
        ply++;

        if (ply > 500) return 0;
    }
}

int main(int argc, char** argv) {
    std::string model1_path;
    std::string model2_path;
    std::string tb_dir = "/home/alvaro/claude/damas";
    int num_pairs = 50;  // 50 pairs = 100 games
    std::uint64_t max_nodes = 10000;
    int random_plies = 10;
    int num_threads = omp_get_max_threads();

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
                      << "  --threads N         Number of threads (default: max available)\n";
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
        } else if (arg == "--threads" && i + 1 < argc) {
            num_threads = std::atoi(argv[++i]);
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

    if (!std::ifstream(model1_path)) {
        std::cerr << "Error: Cannot open model file: " << model1_path << "\n";
        return 1;
    }
    if (!std::ifstream(model2_path)) {
        std::cerr << "Error: Cannot open model file: " << model2_path << "\n";
        return 1;
    }

    omp_set_num_threads(num_threads);

    std::cout << "=== Neural Network Match ===\n";
    std::cout << "Model 1: " << model1_path << "\n";
    std::cout << "Model 2: " << model2_path << "\n";
    std::cout << "Game pairs: " << num_pairs << " (" << (num_pairs * 2) << " games)\n";
    std::cout << "Random opening plies: " << random_plies << "\n";
    std::cout << "Node limit: " << max_nodes << "\n";
    std::cout << "Threads: " << num_threads << "\n";
    std::cout << "Tablebases: " << (tb_dir.empty() ? "disabled" : tb_dir) << "\n\n";

    // Pre-generate all openings (deterministic, single-threaded)
    auto now = std::chrono::high_resolution_clock::now();
    auto seed = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count());
    RandomBits rng(seed);

    std::vector<Board> openings(num_pairs);
    for (int i = 0; i < num_pairs; ++i) {
        openings[i] = generate_opening(rng, random_plies);
    }

    // Preload compressed WDL tablebases (shared read-only across threads)
    CompressedTablebaseManager tb_wdl(tb_dir);
    int tb_piece_limit = 7;
    if (!tb_dir.empty()) {
        tb_wdl.preload(tb_piece_limit);
    }
    // Const pointer ensures thread-safe read-only access
    const CompressedTablebaseManager* tb_ptr = &tb_wdl;

    // Atomic match statistics
    std::atomic<int> model1_wins{0}, model2_wins{0}, draws{0};
    std::atomic<int> model1_white_wins{0}, model1_black_wins{0};
    std::atomic<int> model2_white_wins{0}, model2_black_wins{0};
    std::atomic<int> completed_games{0};
    int total_games = num_pairs * 2;

    std::mutex io_mutex;

    #pragma omp parallel
    {
        // Each thread gets its own searcher pair
        search::Searcher searcher1("", 0, model1_path);
        search::Searcher searcher2("", 0, model2_path);
        searcher1.set_tt_size(32);
        searcher2.set_tt_size(32);

        // WDL probe using const pointer to preloaded tables
        if (!tb_dir.empty()) {
            auto wdl_probe = [tb_ptr](const Board& board) -> tablebase::DTM {
                Value wdl = tb_ptr->lookup_wdl_preloaded(board);
                if (wdl == Value::WIN) return dtm_win(1);
                if (wdl == Value::LOSS) return dtm_loss(1);
                if (wdl == Value::DRAW) return DTM_DRAW;
                return DTM_UNKNOWN;
            };
            searcher1.set_dtm_probe(wdl_probe, tb_piece_limit);
            searcher2.set_dtm_probe(wdl_probe, tb_piece_limit);
        }

        #pragma omp for schedule(dynamic)
        for (int pair = 0; pair < num_pairs; ++pair) {
            const Board& opening = openings[pair];

            for (int swap = 0; swap < 2; ++swap) {
                bool model1_is_white = (swap == 0);

                search::Searcher& white = model1_is_white ? searcher1 : searcher2;
                search::Searcher& black = model1_is_white ? searcher2 : searcher1;

                searcher1.clear_tt();
                searcher2.clear_tt();

                int result = play_game(opening, white, black, max_nodes);

                // Update atomic statistics
                if (result == +1) {
                    if (model1_is_white) { model1_wins++; model1_white_wins++; }
                    else { model2_wins++; model2_white_wins++; }
                } else if (result == -1) {
                    if (model1_is_white) { model2_wins++; model2_black_wins++; }
                    else { model1_wins++; model1_black_wins++; }
                } else {
                    draws++;
                }

                int done = completed_games.fetch_add(1, std::memory_order_relaxed) + 1;

                // Progress every 10 games
                if (done % 10 == 0 || done == total_games) {
                    std::lock_guard<std::mutex> lock(io_mutex);
                    int m1w = model1_wins.load(std::memory_order_relaxed);
                    int m2w = model2_wins.load(std::memory_order_relaxed);
                    int d = draws.load(std::memory_order_relaxed);
                    std::cout << "Progress: " << done << "/" << total_games
                              << "  Score: " << m1w << "-" << m2w << " =" << d << "\n"
                              << std::flush;
                }
            }
        }
    }

    // Final results
    std::cout << "\n=== Final Results ===\n";
    std::cout << "Model 1: " << model1_wins.load() << " wins ("
              << model1_white_wins.load() << " as white, "
              << model1_black_wins.load() << " as black)\n";
    std::cout << "Model 2: " << model2_wins.load() << " wins ("
              << model2_white_wins.load() << " as white, "
              << model2_black_wins.load() << " as black)\n";
    std::cout << "Draws: " << draws.load() << "\n";

    double model1_score = model1_wins.load() + 0.5 * draws.load();
    double model2_score = model2_wins.load() + 0.5 * draws.load();

    std::cout << "\nScore: " << std::fixed << std::setprecision(1)
              << model1_score << " - " << model2_score
              << " (" << (100.0 * model1_score / total_games) << "% - "
              << (100.0 * model2_score / total_games) << "%)\n";

    return 0;
}
