// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/autodiff/Variable.h>
#include <suboptimal/solvers/newton/Newton.h>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Eigen;
using namespace suboptimal;

TEST_CASE("Newton - Single-variable", "[newton]") {
  const double expected_x = std::pow(2, 1.0 / 3.0);

  NonlinearProblem problem{};
  const auto x = problem.makeDecisionVariable();
  problem.minimize(0.05 * suboptimal::pow(x, 4) - 0.4 * x + 1);
  ExitStatus status = solveNewton(problem);

  REQUIRE(status == ExitStatus::Success);
  CHECK_THAT(x.getValue(), Catch::Matchers::WithinAbs(expected_x, 1e-9));
}

TEST_CASE("Newton - Two-variable", "[newton]") {
  constexpr double expected_x = 1.0;
  constexpr double expected_y = 1.0;

  NonlinearProblem problem{};
  const auto x = problem.makeDecisionVariable();
  const auto y = problem.makeDecisionVariable();
  problem.minimize(suboptimal::pow(1 - x, 4) + 100 * suboptimal::pow(y - x * x, 2));

  auto status = solveNewton(problem);

  REQUIRE(status == ExitStatus::Success);
  CHECK_THAT(x.getValue(), Catch::Matchers::WithinAbs(expected_x, 0.0001));
  CHECK_THAT(y.getValue(), Catch::Matchers::WithinAbs(expected_y, 0.0001));
}
