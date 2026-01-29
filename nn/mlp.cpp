#include "mlp.hpp"
#include <immintrin.h>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace nn {

MLP::MLP(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open model file: " + path);
    }

    std::uint32_t num_layers;
    file.read(reinterpret_cast<char*>(&num_layers), sizeof(num_layers));
    if (!file) {
        throw std::runtime_error("Failed to read number of layers");
    }

    layers_.resize(num_layers);
    for (std::uint32_t i = 0; i < num_layers; ++i) {
        auto& layer = layers_[i];

        file.read(reinterpret_cast<char*>(&layer.in_size), sizeof(layer.in_size));
        file.read(reinterpret_cast<char*>(&layer.out_size), sizeof(layer.out_size));
        if (!file) {
            throw std::runtime_error("Failed to read layer dimensions");
        }

        std::size_t weight_count = layer.out_size * layer.in_size;
        layer.weights.resize(weight_count);
        layer.biases.resize(layer.out_size);

        file.read(reinterpret_cast<char*>(layer.weights.data()),
                  weight_count * sizeof(float));
        file.read(reinterpret_cast<char*>(layer.biases.data()),
                  layer.out_size * sizeof(float));

        if (!file) {
            throw std::runtime_error("Failed to read layer weights/biases");
        }
    }
}

std::size_t MLP::num_parameters() const {
    std::size_t total = 0;
    for (const auto& layer : layers_) {
        total += layer.weights.size() + layer.biases.size();
    }
    return total;
}

void MLP::board_to_features(const Board& board, float* features) const {
    // Zero out features
    for (int i = 0; i < 128; ++i) {
        features[i] = 0.0f;
    }

    std::uint32_t white = board.white;
    std::uint32_t black = board.black;
    std::uint32_t kings = board.kings;

    std::uint32_t white_men = white & ~kings;
    std::uint32_t white_kings = white & kings;
    std::uint32_t black_men = black & ~kings;
    std::uint32_t black_kings = black & kings;

    // Extract bits to features
    for (int i = 0; i < 32; ++i) {
        std::uint32_t mask = 1u << i;
        if (white_men & mask)   features[i] = 1.0f;
        if (white_kings & mask) features[32 + i] = 1.0f;
        if (black_men & mask)   features[64 + i] = 1.0f;
        if (black_kings & mask) features[96 + i] = 1.0f;
    }
}

void MLP::matvec_avx(const float* weights, const float* bias,
                     const float* input, float* output,
                     std::uint32_t in_size, std::uint32_t out_size) {
    // Process 8 output elements at a time when possible
    std::uint32_t out_vec = out_size & ~7u;

    for (std::uint32_t o = 0; o < out_vec; o += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        // Process input in chunks of 8
        std::uint32_t in_vec = in_size & ~7u;
        for (std::uint32_t i = 0; i < in_vec; i += 8) {
            __m256 in_v = _mm256_loadu_ps(&input[i]);

            // Each row of weights
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 0) * in_size + i]), in_v, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 1) * in_size + i]), in_v, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 2) * in_size + i]), in_v, acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 3) * in_size + i]), in_v, acc3);
            acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 4) * in_size + i]), in_v, acc4);
            acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 5) * in_size + i]), in_v, acc5);
            acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 6) * in_size + i]), in_v, acc6);
            acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(&weights[(o + 7) * in_size + i]), in_v, acc7);
        }

        // Horizontal sum for each accumulator
        auto hsum = [](__m256 v) -> float {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            lo = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_movehdup_ps(lo);
            lo = _mm_add_ps(lo, shuf);
            shuf = _mm_movehl_ps(shuf, lo);
            lo = _mm_add_ss(lo, shuf);
            return _mm_cvtss_f32(lo);
        };

        float sums[8] = {
            hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3),
            hsum(acc4), hsum(acc5), hsum(acc6), hsum(acc7)
        };

        // Handle remaining input elements
        for (std::uint32_t i = in_vec; i < in_size; ++i) {
            float in_val = input[i];
            for (int k = 0; k < 8; ++k) {
                sums[k] += weights[(o + k) * in_size + i] * in_val;
            }
        }

        // Add bias and store
        for (int k = 0; k < 8; ++k) {
            output[o + k] = sums[k] + bias[o + k];
        }
    }

    // Handle remaining output elements
    for (std::uint32_t o = out_vec; o < out_size; ++o) {
        __m256 acc = _mm256_setzero_ps();

        std::uint32_t in_vec = in_size & ~7u;
        for (std::uint32_t i = 0; i < in_vec; i += 8) {
            __m256 w = _mm256_loadu_ps(&weights[o * in_size + i]);
            __m256 in_v = _mm256_loadu_ps(&input[i]);
            acc = _mm256_fmadd_ps(w, in_v, acc);
        }

        // Horizontal sum
        __m128 lo = _mm256_castps256_ps128(acc);
        __m128 hi = _mm256_extractf128_ps(acc, 1);
        lo = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, lo);
        lo = _mm_add_ss(lo, shuf);
        float sum = _mm_cvtss_f32(lo);

        // Handle remaining input elements
        for (std::uint32_t i = in_vec; i < in_size; ++i) {
            sum += weights[o * in_size + i] * input[i];
        }

        output[o] = sum + bias[o];
    }
}

void MLP::relu_avx(float* data, std::uint32_t size) {
    __m256 zero = _mm256_setzero_ps();

    std::uint32_t vec_size = size & ~7u;
    for (std::uint32_t i = 0; i < vec_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(&data[i], v);
    }

    for (std::uint32_t i = vec_size; i < size; ++i) {
        if (data[i] < 0.0f) data[i] = 0.0f;
    }
}

void MLP::softmax(const float* input, float* output, std::uint32_t size) {
    // Find max for numerical stability
    float max_val = input[0];
    for (std::uint32_t i = 1; i < size; ++i) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (std::uint32_t i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (std::uint32_t i = 0; i < size; ++i) {
        output[i] *= inv_sum;
    }
}

void MLP::forward(const float* input, float* output) const {
    // Allocate buffers for intermediate results
    std::uint32_t max_size = 128;  // Input size
    for (const auto& layer : layers_) {
        if (layer.out_size > max_size) max_size = layer.out_size;
    }

    // Double buffer for in-place computation
    std::vector<float> buf1(max_size);
    std::vector<float> buf2(max_size);

    // Copy input to first buffer
    for (std::uint32_t i = 0; i < 128; ++i) {
        buf1[i] = input[i];
    }

    float* current = buf1.data();
    float* next = buf2.data();

    for (std::size_t l = 0; l < layers_.size(); ++l) {
        const auto& layer = layers_[l];

        matvec_avx(layer.weights.data(), layer.biases.data(),
                   current, next, layer.in_size, layer.out_size);

        // Apply ReLU for all but last layer
        if (l < layers_.size() - 1) {
            relu_avx(next, layer.out_size);
        }

        std::swap(current, next);
    }

    // Copy final layer output
    const auto& last = layers_.back();
    for (std::uint32_t i = 0; i < last.out_size; ++i) {
        output[i] = current[i];
    }
}

void MLP::predict_proba(const Board& board, float& p_loss, float& p_draw, float& p_win) const {
    alignas(32) float features[128];
    board_to_features(board, features);

    float logits[3];
    forward(features, logits);

    float probs[3];
    softmax(logits, probs, 3);

    p_loss = probs[0];
    p_draw = probs[1];
    p_win = probs[2];
}

int MLP::evaluate(const Board& board) const {
    float p_loss, p_draw, p_win;
    predict_proba(board, p_loss, p_draw, p_win);

    // Convert to centipawn-like score: expected value scaled
    // P(win) - P(loss) gives range [-1, 1], scale to [-10000, 10000]
    float expected = p_win - p_loss;
    return static_cast<int>(expected * 10000.0f);
}

} // namespace nn
