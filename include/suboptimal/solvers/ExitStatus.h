// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <array>
#include <string_view>

namespace suboptimal {
enum class ExitStatus : int {
  // The solver found an optimal solution
  Success,
  // The solver determined the problem to be infeasible and stopped
  Infeasible,
  // The solver determined the problem to be unbounded and stopped
  Unbounded,
  // The solver stopped because the iterates were diverging
  DivergingIterates,
  // The solver exceeded the maximum number of iterations without finding a solution
  MaxIterationsExceeded,
  // The solver exceeded the maximum elapsed time without finding a solution
  Timeout
};

constexpr std::string_view toString(const ExitStatus& status) {
  constexpr std::array<std::string_view, 6> strings{"solver found an optimal solution", "problem is infeasible",
                                                    "problem is unbounded",
                                                    "solver detected diverging iterates and stopped"
                                                    "solver exceeded the maximum number of iterations",
                                                    "solver exceeded the maximum elapsed time"};
  return strings[static_cast<int>(status)];
}
}  // namespace suboptimal
