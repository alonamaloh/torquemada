#pragma once
#include <cstdint>

// Software polyfill for x86 BMI2 instructions (pext/pdep) for WASM builds

// Parallel bits extract: extracts bits from src at positions specified by mask
inline std::uint32_t _pext_u32(std::uint32_t src, std::uint32_t mask) {
  std::uint32_t result = 0;
  std::uint32_t dest_bit = 1;
  while (mask) {
    std::uint32_t lowest_bit = mask & -mask;  // isolate lowest set bit
    if (src & lowest_bit) {
      result |= dest_bit;
    }
    dest_bit <<= 1;
    mask &= mask - 1;  // clear lowest set bit
  }
  return result;
}

// Parallel bits deposit: deposits bits from src at positions specified by mask
inline std::uint32_t _pdep_u32(std::uint32_t src, std::uint32_t mask) {
  std::uint32_t result = 0;
  std::uint32_t src_bit = 1;
  while (mask) {
    std::uint32_t lowest_bit = mask & -mask;  // isolate lowest set bit
    if (src & src_bit) {
      result |= lowest_bit;
    }
    src_bit <<= 1;
    mask &= mask - 1;  // clear lowest set bit
  }
  return result;
}
