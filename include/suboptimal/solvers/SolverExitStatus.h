// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <string>

namespace suboptimal {
enum class SolverExitStatus {
  // The solver found an optimal solution
  Success,
  // The solver determined the problem to be infeasible and stopped
  Infeasible,
  // The solver determined the problem to be unbounded and stopped
  Unbounded,
  // The solver exceeded the maximum number of iterations without finding a solution
  MaxIterationsExceeded,
  // The solver exceeded the maximum elapsed time without finding a solution
  Timeout
};

constexpr std::string toString(const SolverExitStatus& status) {
  using enum SolverExitStatus;
  switch (status) {
    case Success:
      return "solver found an optimal solution";
    case Infeasible:
      return "problem is infeasible";
    case Unbounded:
      return "problem is unbounded";
    case MaxIterationsExceeded:
      return "solver exceeded the maximum number of iterations";
    case Timeout:
      return "solver exceeded the maximum elapsed time";
    default:
      return "unknown status";
  }
}
}  // namespace suboptimal
