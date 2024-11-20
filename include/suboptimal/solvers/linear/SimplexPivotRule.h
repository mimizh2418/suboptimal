// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <string>

namespace suboptimal {
enum class SimplexPivotRule {
  // Simplest pivot rule, practical in some cases but may cause cycling
  Dantzig,
  // Prevents cycling, but may take longer to converge
  Bland,
  // Prevents cycling and is more efficient than Bland's rule
  Lexicographic
};

constexpr std::string toString(const SimplexPivotRule& rule) {
  using enum SimplexPivotRule;
  switch (rule) {
    case Dantzig:
      return "Dantzig";
    case Bland:
      return "Bland";
    case Lexicographic:
      return "Lexicographic";
    default:
      return "unknown pivot rule";
  }
}
}  // namespace suboptimal
