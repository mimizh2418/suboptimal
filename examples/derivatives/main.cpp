// Copyright (c) 2024 Alvin Zhang.

#include <suboptimal/autodiff/Variable.h>
#include <suboptimal/autodiff/derivatives.h>

#include <array>
#include <cmath>
#include <format>
#include <iostream>

int main() {
  suboptimal::Variable x{};
  const suboptimal::Variable y = 1.0 / (1.0 + suboptimal::exp(-x));  // Sigmoid function
  suboptimal::Derivative dydx{y, x};

  constexpr double min = -1.0;
  constexpr double max = 1.0;
  constexpr double step = 0.1;

  std::array<double, static_cast<int>(max - min / step) + 1> x_vals{};
  for (unsigned int i = 0; i < x_vals.size(); i++) {
    x_vals[i] = min + i * step;
  }

  std::cout << std::format("Differentiating y = 1 / (1 + e^(-x)), on x ∈ [{}, {}]\n", min, max) << std::endl;

  std::cout << std::format("{:^10}|{:^10}|{:^20}|{:^20}\n", "x", "y", "dy/dx (manual)", "dy/dx (autodiff)")
            << std::format("{:=<63}", "") << std::endl;
  for (double x_val : x_vals) {
    x.setValue(x_val);
    double dydx_manual = std::exp(-x_val) / std::pow(1.0 + std::exp(-x_val), 2);
    double dydx_ad = dydx.getValue();
    std::cout << std::format("{:^10.4f}|{:^10.4f}|{:^20.4f}|{:^20.4f}", x_val, y.getValue(), dydx_manual, dydx_ad)
              << std::endl;

    if (std::abs(dydx_manual - dydx_ad) > 1e-9) {
      std::cerr << std::format(
          "Error: autodiff result is incorrect\n "
          "Manual: {}\n "
          "Autodiff: {}\n",
          dydx_manual, dydx_ad);
      return 1;
    }
  }
  return 0;
}
