#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/derivatives.h>

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
  const Variable f = x * x + 2 * x * y + y * y;
  const Vector2v grad = gradient(f, Vector2v{x, y});

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  const double y_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  y.setValue(y_val);
  CHECK(suboptimal::getValues(grad).isApprox(Eigen::Vector2d{2 * (x_val + y_val), 2 * (x_val + y_val)}));
}

TEST_CASE("Autodiff - Basic derivative", "[autodiff]") {
  Variable x{};
  const Variable y = x * x + 2 * x + 1;
  const Variable dydx = derivative(y, x);

  const double x_val = GENERATE(take(10, random(-100.0, 100.0)));
  x.setValue(x_val);
  CHECK_THAT(dydx.getValue(), Catch::Matchers::WithinAbs(2 * x_val + 2, 1e-9));
}