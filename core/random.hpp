#pragma once
#include <cstdint>
#include <limits>
#include <xmmintrin.h>

inline std::uint64_t avalanche(std::uint64_t x) {
  unsigned __int128 tmp;
  tmp = (unsigned __int128)x * 0xa3b195354a39b70dull;
  x = (tmp >> 64) ^ tmp;
  tmp = (unsigned __int128)x * 0x1b03738712fad5c9ull;
  x = (tmp >> 64) ^ tmp;
  return x;
}

struct RandomBits {
  using result_type = std::uint64_t;

  static constexpr result_type min() { return 0; }

  static constexpr result_type max() {
    return std::numeric_limits<result_type>::max();
  }

  std::uint64_t state;

  RandomBits(std::uint64_t seed)
      : state(avalanche(seed ^ 0xc41cbd5c71679d51ull)) {}

  std::uint64_t operator()() {
    std::uint64_t result = avalanche(state);
    state += 0x60bee2bee120fc15ull;
    return result;
  }
};
