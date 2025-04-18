// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/LinearProblem.h>
#include <suboptimal/solvers/ExitStatus.h>
#include <suboptimal/solvers/simplex/Simplex.h>
#include <suboptimal/solvers/simplex/SimplexConfig.h>
#include <suboptimal/solvers/simplex/SimplexPivotRule.h>

#include <limits>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using namespace Eigen;
using namespace suboptimal;

TEST_CASE("Simplex failure mode - Degenerate cycling", "[simplex]") {
  LinearProblem problem{};
  problem.addLessThanConstraint(Vector4d{{0.5, -5.5, -2.5, 9}}, 0);
  problem.addLessThanConstraint(Vector4d{{0.5, -1.5, -0.5, 1}}, 0);
  problem.addLessThanConstraint(Vector4d{{1, 1, 1, 1}}, 1);
  problem.maximize(Vector4d{{10, -57, -9, -24}});
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  Vector4d solution;
  double objective_value;

  SECTION("Max iterations exceeded") {
    auto status =
        solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = SimplexPivotRule::Dantzig});
    REQUIRE(status == ExitStatus::MaxIterationsExceeded);
  }

  SECTION("Timeout") {
    auto status = solveSimplex(problem, solution, objective_value,
                               {.verbose = true,
                                .max_iterations = std::numeric_limits<int>::max(),
                                .timeout = std::chrono::duration<double, std::milli>{100},
                                .pivot_rule = SimplexPivotRule::Dantzig});
    REQUIRE(status == ExitStatus::Timeout);
  }
}

TEST_CASE("Simplex failure mode - Unbounded problem", "[simplex]") {
  const auto pivot_rule = GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Dantzig, SimplexPivotRule::Bland);

  LinearProblem problem{};
  problem.addLessThanConstraint(Vector3d{{1, -1, 1}}, 5);
  problem.addLessThanConstraint(Vector3d{{-2, 1, 0}}, 3);
  problem.addLessThanConstraint(Vector3d{{0, 1, -2}}, 5);
  problem.maximize(Vector3d{{0, 2, 1}});
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  Vector3d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});

  REQUIRE(status == ExitStatus::Unbounded);
}

TEST_CASE("Simplex failure mode - Infeasible problem", "[simplex]") {
  const auto pivot_rule = GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Dantzig, SimplexPivotRule::Bland);

  LinearProblem problem{};
  problem.addLessThanConstraint(Vector3d{{2, -1, -2}}, 4);
  problem.addGreaterThanConstraint(Vector3d{{-2, 3, 1}}, 5);
  problem.addGreaterThanConstraint(Vector3d{{1, -1, -1}}, 1);
  problem.maximize(Vector3d{{1, -1, 1}});
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 2);

  Vector3d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});

  REQUIRE(status == ExitStatus::Infeasible);
}
