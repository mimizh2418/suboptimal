// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/Variable.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace suboptimal;

TEST_CASE("Autodiff - Variable constructor", "[autodiff]") {
  const Variable x{1.0};
  CHECK(x.getValue() == 1.0);
  CHECK(x.getLinearity() == Linearity::Linear);

  const Variable y{};
  CHECK(y.getValue() == 0.0);
  CHECK(y.getLinearity() == Linearity::Linear);
}

TEST_CASE("Autodiff - Variable basic arithmetic", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable f{};

  SECTION("Initial value") {
    const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
    const double y_val = GENERATE(take(5, random(-100.0, 100.0)));

    x.setValue(x_val);
    y.setValue(y_val);

    f = 1.0 + (x + y) * (x - y) / (x * y);
    CHECK_THAT(f.getValue(),
               Catch::Matchers::WithinAbs(1.0 + (x_val + y_val) * (x_val - y_val) / (x_val * y_val), 1e-9));
  }

  SECTION("Value update") {
    f = 1.0 + (x + y) * (x - y) / (x * y);

    const double x_val = GENERATE(take(5, random(-100.0, 100.0)));
    const double y_val = GENERATE(take(5, random(-100.0, 100.0)));

    x.setValue(x_val);
    y.setValue(y_val);

    CHECK_THAT(f.getValue(),
               Catch::Matchers::WithinAbs(1.0 + (x_val + y_val) * (x_val - y_val) / (x_val * y_val), 1e-9));
  }
}

TEST_CASE("Autodiff - Variable STL functions", "[autodiff]") {
  Variable x{};
  Variable y{};
  Variable f{};

  SECTION("Initial value") {
    const double x_val = GENERATE(take(5, random(0.0, 100.0)));
    const double y_val = GENERATE(take(5, random(0.0, 100.0)));

    x.setValue(x_val);
    y.setValue(y_val);

    f = suboptimal::sin(x) + suboptimal::cos(y) +
        suboptimal::atan2(x, 5) * suboptimal::sqrt(y) / (suboptimal::exp(x) + suboptimal::hypot(suboptimal::log(x), y));
    const double f_val =
        std::sin(x_val) + std::cos(y_val) +
        std::atan2(x_val, 5) * std::sqrt(y_val) / (std::exp(x_val) + std::hypot(std::log(x_val), y_val));
    CHECK_THAT(f.getValue(), Catch::Matchers::WithinAbs(f_val, 1e-9));
  }

  SECTION("Value update") {
    f = suboptimal::sin(x) + suboptimal::cos(y) +
        suboptimal::atan2(x, 5) * suboptimal::sqrt(y) / (suboptimal::exp(x) + suboptimal::hypot(suboptimal::log(x), y));

    const double x_val = GENERATE(take(5, random(0.0, 100.0)));
    const double y_val = GENERATE(take(5, random(0.0, 100.0)));

    x.setValue(x_val);
    y.setValue(y_val);

    const double f_val =
        std::sin(x_val) + std::cos(y_val) +
        std::atan2(x_val, 5) * std::sqrt(y_val) / (std::exp(x_val) + std::hypot(std::log(x_val), y_val));
    CHECK_THAT(f.getValue(), Catch::Matchers::WithinAbs(f_val, 1e-9));
  }
}

TEST_CASE("Autodiff - Eigen Variable support", "[autodiff]") {
  SECTION("Matrix operations") {
    const Eigen::Matrix2d x_val{{1, 2},  //
                                {3, 4}};
    const Eigen::Matrix2d y_val{{5, 6},  //
                                {7, 8}};
    const Eigen::Matrix2d f_val = x_val * y_val;

    const Matrix2v x{{Variable{1}, Variable{2}},  //
                     {Variable{3}, Variable{4}}};
    const Matrix2v y{{Variable{5}, Variable{6}},  //
                     {Variable{7}, Variable{8}}};
    const Matrix2v f = x * y;
    CHECK(suboptimal::getValues(f).isApprox(f_val));
    CHECK(suboptimal::getValuesSparse(f).isApprox(f_val));
  }

  SECTION("Vector operations") {
    const Eigen::Vector4d x_val{1, 2, 3, 4};
    const Eigen::Vector4d y_val{5, 6, 7, 8};
    const double f_val = x_val.dot(y_val);

    const Vector4v x{Variable{1}, Variable{2}, Variable{3}, Variable{4}};
    const Vector4v y{Variable{5}, Variable{6}, Variable{7}, Variable{8}};
    const Variable f = x.dot(y);

    CHECK_THAT(f.getValue(), Catch::Matchers::WithinAbs(f_val, 1e-9));
  }
}
