// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <string>

namespace suboptimal {
enum class SolverExitStatus {
  // The solver found an optimal solution
  kSuccess,
  // The solver determined the problem to be infeasible and stopped
  kInfeasible,
  // The solver determined the problem to be unbounded and stopped
  kUnbounded,
  // The solver exceeded the maximum number of iterations without finding a solution
  kMaxIterationsExceeded,
  // The solver exceeded the maximum elapsed time without finding a solution
  kTimeout
};

constexpr std::string toString(const SolverExitStatus& status) {
  using enum SolverExitStatus;
  switch (status) {
    case kSuccess:
      return "solver found an optimal solution";
    case kInfeasible:
      return "problem is infeasible";
    case kUnbounded:
      return "problem is unbounded";
    case kMaxIterationsExceeded:
      return "solver exceeded the maximum number of iterations";
    case kTimeout:
      return "solver exceeded the maximum elapsed time";
    default:
      return "unknown status";
  }
}
}  // namespace suboptimal
