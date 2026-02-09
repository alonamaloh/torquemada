#include "mlp.hpp"

#ifdef WASM_BUILD
#include <wasm_simd128.h>
#else
#include <immintrin.h>
#endif

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

    // Compute max layer size for stack allocation in forward()
    max_layer_size_ = 128;  // Input size
    for (const auto& layer : layers_) {
        if (layer.out_size > max_layer_size_) max_layer_size_ = layer.out_size;
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

#ifdef WASM_BUILD

// WASM SIMD128 implementation (128-bit vectors, 4 floats at a time)
void MLP::matvec_avx(const float* weights, const float* bias,
                     const float* input, float* output,
                     std::uint32_t in_size, std::uint32_t out_size) {
    // Process 4 output elements at a time
    std::uint32_t out_vec = out_size & ~3u;

    for (std::uint32_t o = 0; o < out_vec; o += 4) {
        v128_t acc0 = wasm_f32x4_splat(0.0f);
        v128_t acc1 = wasm_f32x4_splat(0.0f);
        v128_t acc2 = wasm_f32x4_splat(0.0f);
        v128_t acc3 = wasm_f32x4_splat(0.0f);

        // Process input in chunks of 4
        std::uint32_t in_vec = in_size & ~3u;
        for (std::uint32_t i = 0; i < in_vec; i += 4) {
            v128_t in_v = wasm_v128_load(&input[i]);

            // Each row of weights - use fma (multiply-add)
            acc0 = wasm_f32x4_add(acc0, wasm_f32x4_mul(wasm_v128_load(&weights[(o + 0) * in_size + i]), in_v));
            acc1 = wasm_f32x4_add(acc1, wasm_f32x4_mul(wasm_v128_load(&weights[(o + 1) * in_size + i]), in_v));
            acc2 = wasm_f32x4_add(acc2, wasm_f32x4_mul(wasm_v128_load(&weights[(o + 2) * in_size + i]), in_v));
            acc3 = wasm_f32x4_add(acc3, wasm_f32x4_mul(wasm_v128_load(&weights[(o + 3) * in_size + i]), in_v));
        }

        // Horizontal sum for each accumulator
        auto hsum = [](v128_t v) -> float {
            // v = [a, b, c, d]
            // Step 1: shuffle to get [c, d, ?, ?] and add
            v128_t hi = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);  // [c, d, a, b]
            v = wasm_f32x4_add(v, hi);  // [a+c, b+d, ?, ?]
            // Step 2: shuffle to get [b+d, ?, ?, ?] and add
            hi = wasm_i32x4_shuffle(v, v, 1, 0, 3, 2);  // [b+d, a+c, ?, ?]
            v = wasm_f32x4_add(v, hi);  // [a+b+c+d, ?, ?, ?]
            return wasm_f32x4_extract_lane(v, 0);
        };

        float sums[4] = {
            hsum(acc0), hsum(acc1), hsum(acc2), hsum(acc3)
        };

        // Handle remaining input elements
        for (std::uint32_t i = in_vec; i < in_size; ++i) {
            float in_val = input[i];
            for (int k = 0; k < 4; ++k) {
                sums[k] += weights[(o + k) * in_size + i] * in_val;
            }
        }

        // Add bias and store
        for (int k = 0; k < 4; ++k) {
            output[o + k] = sums[k] + bias[o + k];
        }
    }

    // Handle remaining output elements (scalar)
    for (std::uint32_t o = out_vec; o < out_size; ++o) {
        v128_t acc = wasm_f32x4_splat(0.0f);

        std::uint32_t in_vec = in_size & ~3u;
        for (std::uint32_t i = 0; i < in_vec; i += 4) {
            v128_t w = wasm_v128_load(&weights[o * in_size + i]);
            v128_t in_v = wasm_v128_load(&input[i]);
            acc = wasm_f32x4_add(acc, wasm_f32x4_mul(w, in_v));
        }

        // Horizontal sum
        v128_t hi = wasm_i32x4_shuffle(acc, acc, 2, 3, 0, 1);
        acc = wasm_f32x4_add(acc, hi);
        hi = wasm_i32x4_shuffle(acc, acc, 1, 0, 3, 2);
        acc = wasm_f32x4_add(acc, hi);
        float sum = wasm_f32x4_extract_lane(acc, 0);

        // Handle remaining input elements
        for (std::uint32_t i = in_vec; i < in_size; ++i) {
            sum += weights[o * in_size + i] * input[i];
        }

        output[o] = sum + bias[o];
    }
}

#else

// AVX2 implementation (256-bit vectors, 8 floats at a time)
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

#endif // WASM_BUILD

void MLP::relu_avx(float* data, std::uint32_t size) {
#ifdef WASM_BUILD
    v128_t zero = wasm_f32x4_splat(0.0f);

    std::uint32_t vec_size = size & ~3u;
    for (std::uint32_t i = 0; i < vec_size; i += 4) {
        v128_t v = wasm_v128_load(&data[i]);
        v = wasm_f32x4_max(v, zero);
        wasm_v128_store(&data[i], v);
    }
#else
    __m256 zero = _mm256_setzero_ps();

    std::uint32_t vec_size = size & ~7u;
    for (std::uint32_t i = 0; i < vec_size; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        v = _mm256_max_ps(v, zero);
        _mm256_storeu_ps(&data[i], v);
    }
#endif

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
    // Stack-allocated buffers (no heap allocation, thread-safe)
    alignas(32) float buf1[512];
    alignas(32) float buf2[512];

    // Copy input to first buffer
    for (std::uint32_t i = 0; i < 128; ++i) {
        buf1[i] = input[i];
    }

    float* current = buf1;
    float* next = buf2;

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

int MLP::evaluate(const Board& board, int draw_score) const {
    float p_loss, p_draw, p_win;
    predict_proba(board, p_loss, p_draw, p_win);

    // Weighted average: p_win * 10000 + p_draw * draw_score + p_loss * (-10000)
    float expected = p_win * 10000.0f + p_draw * static_cast<float>(draw_score) + p_loss * (-10000.0f);
    return static_cast<int>(expected);
}

} // namespace nn
