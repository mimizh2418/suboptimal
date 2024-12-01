// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <array>
#include <string_view>

namespace suboptimal {
enum class SimplexPivotRule : int {
  // Simplest pivot rule, practical in some cases but may cause cycling
  Dantzig,
  // Prevents cycling, but may take longer to converge
  Bland,
  // Prevents cycling and is more efficient than Bland's rule
  Lexicographic
};

constexpr std::string_view toString(const SimplexPivotRule& rule) {
  constexpr std::array<std::string_view, 3> strings{"Dantzig", "Bland", "Lexicographic"};
  return strings[static_cast<int>(rule)];
}
}  // namespace suboptimal
