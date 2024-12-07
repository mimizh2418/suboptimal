// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <chrono>
#include <limits>

namespace suboptimal {
struct InteriorPointConfig {
  // Solver error tolerance
  double tolerance = 1e-9;
  // Enables verbose output
  bool verbose = false;
  // Maximum number of iterations before solver gives up
  int max_iterations = 5000;
  // Maximum elapsed time before solver gives up
  std::chrono::duration<double, std::milli> timeout{std::numeric_limits<double>::infinity()};
};
}  // namespace suboptimal
