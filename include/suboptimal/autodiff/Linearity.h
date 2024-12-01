// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <array>
#include <string_view>

namespace suboptimal {
enum class Linearity : int {
  // The expression is a constant
  Constant,
  // The expression contains linear and constant terms
  Linear,
  // The expression contains quadratic terms and lower
  Quadratic,
  // The expression contains nonlinear terms and lower
  Nonlinear
};

constexpr std::string_view toString(const Linearity& linearity) {
  constexpr std::array<std::string_view, 4> strings{"constant", "linear", "quadratic", "nonlinear"};
  return strings[static_cast<int>(linearity)];
}
}  // namespace suboptimal
