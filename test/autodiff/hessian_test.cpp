// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/Variable.h>
#include <suboptimal/autodiff/derivatives.h>

#include <cmath>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

using namespace suboptimal;

TEST_CASE("Autodiff - Basic Hessian") {
  Variable x{};
  Variable y{};
  const Variable f = (x * x - 2 * x * y + y * y);
  Hessian h{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-100, 100)));
  const double y_val = GENERATE(take(10, random(-100, 100)));
  x.setValue(x_val);
  y.setValue(y_val);

  const auto expected_h = Eigen::Matrix2d{{2, -2},  //
                                          {-2, 2}};
  CHECK(h.getValue().isApprox(expected_h));
  CHECK(suboptimal::getValues(h.getExpr()).isApprox(expected_h));
}

// I'm too lazy to write more tests for hessian, but if both gradient and jacobian work hessian should also work
TEST_CASE("Autodiff - Complicated Hessian") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = x * y * z + y * suboptimal::sin(x) + z * suboptimal::cos(y);
  Hessian h{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-100, 100)));
  const double y_val = GENERATE(take(5, random(-100, 100)));
  const double z_val = GENERATE(take(5, random(-100, 100)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const auto expected_h = Eigen::Matrix3d{{-y_val * std::sin(x_val), std::cos(x_val) + z_val, y_val},
                                          {std::cos(x_val) + z_val, -z_val * std::cos(y_val), x_val - std::sin(y_val)},
                                          {y_val, x_val - std::sin(y_val), 0}};
  CHECK(h.getValue().isApprox(expected_h));
  CHECK(suboptimal::getValues(h.getExpr()).isApprox(expected_h));
}
