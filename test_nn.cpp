#include "core/board.hpp"
#include "nn/mlp.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        std::cout << "Usage: " << argv[0] << " <model.bin>\n"
                  << "Test neural network inference\n\n"
                  << "Arguments:\n"
                  << "  model.bin           Neural network model file\n\n"
                  << "Options:\n"
                  << "  -h, --help          Show this help message\n";
        return argc < 2 ? 1 : 0;
    }

    std::cout << "Loading model from " << argv[1] << "...\n";
    nn::MLP model(argv[1]);
    std::cout << "Model has " << model.num_parameters() << " parameters\n\n";

    // Test on starting position
    Board board;
    float p_loss, p_draw, p_win;

    model.predict_proba(board, p_loss, p_draw, p_win);
    int score = model.evaluate(board);

    std::cout << "Starting position:\n";
    std::cout << "  P(loss): " << std::fixed << std::setprecision(3) << p_loss << "\n";
    std::cout << "  P(draw): " << p_draw << "\n";
    std::cout << "  P(win):  " << p_win << "\n";
    std::cout << "  Score:   " << score << "\n\n";

    // Benchmark
    std::cout << "Benchmarking...\n";
    const int iterations = 100000;

    auto start = std::chrono::high_resolution_clock::now();
    int dummy = 0;
    for (int i = 0; i < iterations; ++i) {
        dummy += model.evaluate(board);
    }
    auto end = std::chrono::high_resolution_clock::now();
    (void) dummy;

    double elapsed = std::chrono::duration<double>(end - start).count();
    double evals_per_sec = iterations / elapsed;

    std::cout << "  " << iterations << " evaluations in "
              << std::fixed << std::setprecision(2) << elapsed << " seconds\n";
    std::cout << "  " << std::fixed << std::setprecision(0)
              << evals_per_sec << " evals/sec\n";
    std::cout << "  " << std::fixed << std::setprecision(2)
              << (1e6 / evals_per_sec) << " us/eval\n";

    return 0;
}
