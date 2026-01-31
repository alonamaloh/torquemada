#include "dtm_sampler.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

double DTMSampler::compute_weight(const Material& m) {
  int queens = m.white_queens + m.black_queens;
  if (queens <= 2) return 1.0;
  if (queens <= 4) return 0.5;
  return 0.1;
}

void DTMSampler::load(const std::string& directory) {
  tables_.clear();
  cumulative_weights_.clear();
  total_weight_ = 0;
  total_positions_ = 0;

  // Find all dtm_*.bin files
  std::vector<fs::path> dtm_files;
  for (const auto& entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file()) {
      std::string name = entry.path().filename().string();
      if (name.starts_with("dtm_") && name.ends_with(".bin")) {
        dtm_files.push_back(entry.path());
      }
    }
  }

  std::sort(dtm_files.begin(), dtm_files.end());
  std::cout << "Found " << dtm_files.size() << " DTM files" << std::endl;

  // Load each file
  for (const auto& path : dtm_files) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      std::cerr << "Warning: cannot open " << path << std::endl;
      continue;
    }

    // Read header
    std::uint8_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    Material m;
    file.read(reinterpret_cast<char*>(&m), sizeof(Material));

    std::size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));

    if (!file) {
      std::cerr << "Warning: failed to read header from " << path << std::endl;
      continue;
    }

    // Read DTM data
    std::vector<std::int8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    if (!file) {
      std::cerr << "Warning: failed to read data from " << path << std::endl;
      continue;
    }

    double weight = compute_weight(m);

    tables_.push_back(DTMTable{m, std::move(data), weight});
    total_weight_ += weight * tables_.back().data.size();
    total_positions_ += tables_.back().data.size();

    cumulative_weights_.push_back(total_weight_);
  }

  std::cout << "Loaded " << tables_.size() << " tables, "
            << total_positions_ << " total positions" << std::endl;
}

void DTMSampler::board_to_features(const Board& b, float* out) const {
  Bb white_men = b.white & ~b.kings;
  Bb white_kings = b.white & b.kings;
  Bb black_men = b.black & ~b.kings;
  Bb black_kings = b.black & b.kings;

  for (int i = 0; i < 32; i++) {
    Bb mask = 1u << i;
    out[i] = (white_men & mask) ? 1.0f : 0.0f;
    out[32 + i] = (white_kings & mask) ? 1.0f : 0.0f;
    out[64 + i] = (black_men & mask) ? 1.0f : 0.0f;
    out[96 + i] = (black_kings & mask) ? 1.0f : 0.0f;
  }
}

void DTMSampler::sample_batch(int batch_size, float* features, int* classes) {
  std::uniform_real_distribution<double> table_dist(0.0, total_weight_);

  for (int i = 0; i < batch_size; i++) {
    Board board;
    DTM dtm;

    // Rejection sampling: keep picking until we find a quiet position
    while (true) {
      // Pick a table weighted by (weight * size)
      double r = table_dist(rng_);
      auto it = std::lower_bound(cumulative_weights_.begin(),
                                  cumulative_weights_.end(), r);
      std::size_t table_idx = it - cumulative_weights_.begin();
      if (table_idx >= tables_.size()) table_idx = tables_.size() - 1;

      const DTMTable& table = tables_[table_idx];

      // Pick a random position within the table
      std::uniform_int_distribution<std::size_t> pos_dist(0, table.data.size() - 1);
      std::size_t pos_idx = pos_dist(rng_);

      // Convert index to board
      board = index_to_board(pos_idx, table.material);

      // Only accept quiet positions (no captures available)
      if (!has_captures(board)) {
        dtm = static_cast<DTM>(table.data[pos_idx]);
        break;
      }
    }

    // Convert to features
    board_to_features(board, features + i * 128);

    // Get DTM class
    classes[i] = dtm_to_class(dtm);
  }
}
