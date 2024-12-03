// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/Variable.h>
#include <suboptimal/autodiff/derivatives.h>

#include <cmath>
#include <iostream>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace suboptimal;

TEST_CASE("Autodiff - Basic gradient", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = (x * x - 2 * x * y + y * y) / (5 * x);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{(x_val * x_val - y_val * y_val) / (5 * x_val * x_val),  //
                                 (2 * y_val - 2 * x_val) / (5 * x_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of abs", "[autodiff]") {
  Variable x{};
  const Variable f = suboptimal::abs(x);
  Gradient grad{f, {x}};

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);

  const double grad_val = x_val / std::abs(x_val);
  CHECK_THAT(grad.getValue().coeff(0), Catch::Matchers::WithinAbs(grad_val, 1e-9));
  CHECK_THAT(suboptimal::getValues(grad.getExpr())(0), Catch::Matchers::WithinAbs(x_val / std::abs(x_val), 1e-9));
}

TEST_CASE("Autodiff - Gradient of sqrt", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = suboptimal::sqrt(x + 2 * y - z);
  Gradient grad{f, Vector3v{x, y, z}};

  const double x_val = GENERATE(take(5, random(0.0, 100.0)));
  const double y_val = GENERATE(take(5, random(0.0, 100.0)));
  const double z_val = GENERATE(take(5, random(-100.0, 0.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const Eigen::Vector3d grad_val{0.5 / std::sqrt(x_val + 2 * y_val - z_val),  //
                                 1 / std::sqrt(x_val + 2 * y_val - z_val),    //
                                 -0.5 / std::sqrt(x_val + 2 * y_val - z_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of exp", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = suboptimal::exp(x * y);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-10.0, 10.0)));
  const double y_val = GENERATE(take(10, random(-10.0, 10.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{y_val * std::exp(x_val * y_val),  //
                                 x_val * std::exp(x_val * y_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of log", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = suboptimal::log(x * y);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(0.1, 100.0)));
  const double y_val = GENERATE(take(10, random(0.1, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{1 / x_val,  //
                                 1 / y_val};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of pow", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = suboptimal::pow(x, y);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(0.0, 100.0)));
  const double y_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{y_val * std::pow(x_val, y_val - 1),  //
                                 std::log(x_val) * std::pow(x_val, y_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of hypot", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = suboptimal::hypot(x, y);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{x_val / std::hypot(x_val, y_val),  //
                                 y_val / std::hypot(x_val, y_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of erf", "[autodiff]") {
  Variable x{};
  const Variable f = suboptimal::erf(x);
  Gradient grad{f, {x}};

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  const double grad_val = 2 * std::exp(-x_val * x_val) / std::sqrt(std::numbers::pi);
  CHECK_THAT(grad.getValue().coeff(0), Catch::Matchers::WithinAbs(grad_val, 1e-9));
  CHECK_THAT(suboptimal::getValues(grad.getExpr())(0), Catch::Matchers::WithinAbs(grad_val, 1e-9));
}

TEST_CASE("Autodiff - Gradients of trig functions", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = suboptimal::sin(x) * suboptimal::cos(y) * suboptimal::tan(z);
  Gradient grad{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double z_val = GENERATE(take(5, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const Eigen::Vector3d grad_val{std::cos(x_val) * std::cos(y_val) * std::tan(z_val),   //
                                 -std::sin(x_val) * std::sin(y_val) * std::tan(z_val),  //
                                 std::sin(x_val) * std::cos(y_val) / (std::cos(z_val) * std::cos(z_val))};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradients of inverse trig functions", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = suboptimal::asin(x) * suboptimal::acos(y) * suboptimal::atan(z);
  Gradient grad{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-1.0, 1.0)));
  const double y_val = GENERATE(take(5, random(-1.0, 1.0)));
  const double z_val = GENERATE(take(5, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const Eigen::Vector3d grad_val{1 / std::sqrt(1 - x_val * x_val) * std::acos(y_val) * std::atan(z_val),   //
                                 -1 / std::sqrt(1 - y_val * y_val) * std::asin(x_val) * std::atan(z_val),  //
                                 1 / (1 + z_val * z_val) * std::asin(x_val) * std::acos(y_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of atan2", "[autodiff]") {
  Variable x{};
  Variable y{};
  const Variable f = suboptimal::atan2(y, x);
  Gradient grad{f, {x, y}};

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);

  const Eigen::Vector2d grad_val{-y_val / (x_val * x_val + y_val * y_val),  //
                                 x_val / (x_val * x_val + y_val * y_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of hyperbolic trig functions", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = suboptimal::sinh(x) * suboptimal::cosh(y) * suboptimal::tanh(z);
  Gradient grad{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double z_val = GENERATE(take(5, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const Eigen::Vector3d grad_val{std::cosh(x_val) * std::cosh(y_val) * std::tanh(z_val),  //
                                 std::sinh(x_val) * std::sinh(y_val) * std::tanh(z_val),  //
                                 std::sinh(x_val) * std::cosh(y_val) / (std::cosh(z_val) * std::cosh(z_val))};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}

TEST_CASE("Autodiff - Gradient of inverse hyperbolic trig functions", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable z{};
  const Variable f = suboptimal::asinh(x) * suboptimal::acosh(y) * suboptimal::atanh(z);
  Gradient grad{f, {x, y, z}};

  const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(5, random(1.0, 100.0)));
  const double z_val = GENERATE(take(5, random(-1.0, 1.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  z.setValue(z_val);

  const Eigen::Vector3d grad_val{std::acosh(y_val) * std::atanh(z_val) / std::sqrt(1 + x_val * x_val),
                                 std::asinh(x_val) * std::atanh(z_val) / (std::sqrt(y_val - 1) * std::sqrt(y_val + 1)),
                                 std::asinh(x_val) * std::acosh(y_val) / (1 - z_val * z_val)};
  CHECK(grad.getValue().isApprox(grad_val));
  CHECK(suboptimal::getValues(grad.getExpr()).isApprox(grad_val));
}
