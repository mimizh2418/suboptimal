#pragma once

#include <chrono>
#include <limits>

#include "suboptimal/solvers/linear/SimplexPivotRule.h"

namespace suboptimal {
struct SimplexSolverConfig {
  // Enables verbose output
  bool verbose = false;
  // Maximum number of iterations before solver gives up
  int max_iterations = 5000;
  // Maximum elapsed time before solver gives up
  std::chrono::duration<double> timeout{std::numeric_limits<double>::infinity()};
  // Pivot rule to use
  SimplexPivotRule pivot_rule = SimplexPivotRule::kLexicographic;
};
}  // namespace suboptimal