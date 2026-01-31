#pragma once

#include "tablebase/tablebase.hpp"
#include "core/movegen.hpp"
#include <vector>
#include <random>
#include <string>

// WDL class labels (same as 8+ piece model)
// 0 = LOSS, 1 = DRAW, 2 = WIN
constexpr int WDL_LOSS = 0;
constexpr int WDL_DRAW = 1;
constexpr int WDL_WIN = 2;
constexpr int NUM_WDL_CLASSES = 3;

// Convert DTM value to WDL class
inline int dtm_to_wdl(DTM dtm) {
  if (dtm > 0) return WDL_WIN;
  if (dtm == 0) return WDL_DRAW;
  return WDL_LOSS;  // dtm < 0 (including DTM_LOSS_TERMINAL)
}

// One loaded DTM tablebase
struct DTMTable {
  Material material;
  std::vector<std::int8_t> data;  // Raw DTM values (1 byte each)
  double weight;  // Sampling weight based on queen count
};

// Sampler that loads DTM tablebases and generates training batches
class DTMSampler {
public:
  // Load DTM files from directory
  // min_pieces/max_pieces: only load tables with this piece count range
  void load(const std::string& directory, int min_pieces = 2, int max_pieces = 7);

  // Sample a batch of (features, wdl_classes)
  // Returns: features as flat array [batch_size * 128], classes as [batch_size]
  // Classes are WDL: 0=LOSS, 1=DRAW, 2=WIN
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
