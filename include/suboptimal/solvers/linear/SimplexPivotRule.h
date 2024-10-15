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

inline std::string toString(const SimplexPivotRule& rule) {
  switch (rule) {
    case SimplexPivotRule::kDantzig:
      return "Dantzig";
    case SimplexPivotRule::kBland:
      return "Bland";
    case SimplexPivotRule::kLexicographic:
      return "Lexicographic";
    default:
      return "unknown pivot rule";
  }
}
}  // namespace suboptimal