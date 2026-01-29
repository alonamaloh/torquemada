#pragma once

#include "../core/board.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nn {

// MLP inference for checkers evaluation
class MLP {
public:
    // Load model from binary file
    explicit MLP(const std::string& path);

    // Evaluate a position, returns score in centipawns-like units
    // Positive = good for side to move, negative = bad
    int evaluate(const Board& board) const;

    // Get raw output probabilities [loss, draw, win]
    void predict_proba(const Board& board, float& p_loss, float& p_draw, float& p_win) const;

    // Get number of parameters
    std::size_t num_parameters() const;

private:
    struct Layer {
        std::uint32_t in_size;
        std::uint32_t out_size;
        std::vector<float> weights;  // [out_size * in_size], row-major
        std::vector<float> biases;   // [out_size]
    };

    std::vector<Layer> layers_;

    // Convert board to 128 input features
    void board_to_features(const Board& board, float* features) const;

    // Forward pass through the network
    void forward(const float* input, float* output) const;

    // SIMD-accelerated matrix-vector multiply: out = weights * in + bias
    static void matvec_avx(const float* weights, const float* bias,
                           const float* input, float* output,
                           std::uint32_t in_size, std::uint32_t out_size);

    // ReLU activation in-place
    static void relu_avx(float* data, std::uint32_t size);

    // Softmax for final output
    static void softmax(const float* input, float* output, std::uint32_t size);
};

} // namespace nn
