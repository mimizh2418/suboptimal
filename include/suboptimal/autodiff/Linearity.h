// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <string>

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

constexpr std::string toString(const Linearity& linearity) {
  using enum Linearity;
  switch (linearity) {
    case Constant:
      return "constant";
    case Linear:
      return "linear";
    case Quadratic:
      return "quadratic";
    case Nonlinear:
      return "nonlinear";
    default:
      return "unknown";
  }
}
}  // namespace suboptimal
