#pragma once

#include "tablebase/tablebase.hpp"
#include <vector>
#include <random>
#include <string>

// DTM class labels (15 classes with logarithmic boundaries)
// Classes 0-6: WIN_1, WIN_2_3, WIN_4_7, WIN_8_15, WIN_16_31, WIN_32_63, WIN_64_127
// Class 7: DRAW
// Classes 8-14: LOSS_64_127, LOSS_32_63, LOSS_16_31, LOSS_8_15, LOSS_4_7, LOSS_2_3, LOSS_1
constexpr int NUM_DTM_CLASSES = 15;
constexpr int DTM_CLASS_DRAW = 7;

// Convert DTM value to class index
inline int dtm_to_class(DTM dtm) {
  if (dtm == 0) return DTM_CLASS_DRAW;

  int moves = dtm > 0 ? dtm : -dtm;
  if (dtm == DTM_LOSS_TERMINAL) moves = 1;  // Terminal loss → LOSS_1

  // Logarithmic bucket: find which power of 2 range
  int bucket;
  if (moves == 1) bucket = 0;
  else if (moves <= 3) bucket = 1;
  else if (moves <= 7) bucket = 2;
  else if (moves <= 15) bucket = 3;
  else if (moves <= 31) bucket = 4;
  else if (moves <= 63) bucket = 5;
  else bucket = 6;

  if (dtm > 0) {
    return bucket;  // WIN classes: 0-6
  } else {
    return 14 - bucket;  // LOSS classes: 14-8 (mirror of WIN)
  }
}

// Midpoint of each class for computing expected scores
inline int class_to_dtm_midpoint(int cls) {
  // Bucket midpoints: 1, 2.5, 5.5, 11.5, 23.5, 47.5, 95.5
  static const int midpoints[] = {1, 2, 5, 11, 23, 47, 95};

  if (cls == DTM_CLASS_DRAW) return 0;

  if (cls < DTM_CLASS_DRAW) {
    return midpoints[cls];  // WIN: positive
  } else {
    return -midpoints[14 - cls];  // LOSS: negative
  }
}

// One loaded DTM tablebase
struct DTMTable {
  Material material;
  std::vector<std::int8_t> data;  // Raw DTM values (1 byte each)
  double weight;  // Sampling weight based on queen count
};

// Sampler that loads all DTM tablebases and generates training batches
class DTMSampler {
public:
  // Load all DTM files from directory
  void load(const std::string& directory);

  // Sample a batch of (features, classes)
  // Returns: features as flat array [batch_size * 128], classes as [batch_size]
  void sample_batch(int batch_size, float* features, int* classes);

  // Total number of positions across all tables
  std::size_t total_positions() const { return total_positions_; }

  // Number of loaded tables
  std::size_t num_tables() const { return tables_.size(); }

private:
  std::vector<DTMTable> tables_;
  std::vector<double> cumulative_weights_;
  double total_weight_ = 0;
  std::size_t total_positions_ = 0;
  std::mt19937_64 rng_{std::random_device{}()};

  // Convert board to 128 features (same as Python train_mlp.py)
  void board_to_features(const Board& b, float* out) const;

  // Compute sampling weight for a material configuration
  static double compute_weight(const Material& m);
};
