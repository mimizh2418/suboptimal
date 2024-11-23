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
  CHECK(x.getType() == ExpressionType::Linear);

  const Variable y{};
  CHECK(y.getValue() == 0.0);
  CHECK(y.getType() == ExpressionType::Linear);
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

    f.update();
    CHECK_THAT(f.getValue(),
               Catch::Matchers::WithinAbs(1.0 + (x_val + y_val) * (x_val - y_val) / (x_val * y_val), 1e-9));
  }
}

TEST_CASE("Autodiff - Variable matrix", "[autodiff]") {
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

    for (Eigen::Index i = 0; i < f.reshaped().size(); i++) {
      CHECK_THAT(f.reshaped()(i).getValue(), Catch::Matchers::WithinAbs(f_val(i), 1e-9));
    }
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
