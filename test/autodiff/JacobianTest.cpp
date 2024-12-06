// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/Variable.h>
#include <suboptimal/autodiff/Derivatives.h>

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>

using namespace suboptimal;

TEST_CASE("Autodiff - Basic Jacobian", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Vector2v f{4 * suboptimal::pow(x, 2) * y,  //
                   x - suboptimal::pow(y, 2)};
  Jacobian j{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-100, 100)));
  const double y_val = GENERATE(take(10, random(-100, 100)));
  x.setValue(x_val);
  y.setValue(y_val);

  const auto expected_j = Eigen::Matrix2d{{8 * x_val * y_val, 4 * x_val * x_val},  //
                                          {1, -2 * y_val}};
  CHECK(j.getValue().isApprox(expected_j));
  CHECK(suboptimal::getValues(j.getExpr()).isApprox(expected_j));
}

// I'm too lazy to write more tests for jacobian, but if gradient works jacobian should also work
TEST_CASE("Autodiff - Complicated Jacobian", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Vector3v f{x * y * z,               //
                   y * suboptimal::sin(x),  //
                   z * suboptimal::cos(y)};
  Jacobian j{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double z_val = GENERATE(take(5, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);
  const auto expected_j = Eigen::Matrix3d{{y_val * z_val, x_val * z_val, x_val * y_val},
                                          {y_val * std::cos(x_val), std::sin(x_val), 0},
                                          {0, -z_val * std::sin(y_val), std::cos(y_val)}};
  CHECK(j.getValue().isApprox(expected_j));
  CHECK(suboptimal::getValues(j.getExpr()).isApprox(expected_j));
}
