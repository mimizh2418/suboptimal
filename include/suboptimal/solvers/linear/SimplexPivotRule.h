// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <string>

namespace suboptimal {
enum class SimplexPivotRule {
  // Simplest pivot rule, practical in some cases but may cause cycling
  kDantzig,
  // Prevents cycling, but may take longer to converge
  kBland,
  // Prevents cycling and is more efficient than Bland's rule
  kLexicographic
};

constexpr std::string toString(const SimplexPivotRule& rule) {
  using enum SimplexPivotRule;
  switch (rule) {
    case kDantzig:
      return "Dantzig";
    case kBland:
      return "Bland";
    case kLexicographic:
      return "Lexicographic";
    default:
      return "unknown pivot rule";
  }
}
}  // namespace suboptimal
