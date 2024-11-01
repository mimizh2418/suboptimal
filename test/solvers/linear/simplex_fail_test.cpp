// Copyright (c) 2024 Alvin Zhang.

#include <suboptimal/LinearProblem.h>
#include <suboptimal/solvers/SolverExitStatus.h>
#include <suboptimal/solvers/linear/SimplexPivotRule.h>
#include <suboptimal/solvers/linear/SimplexSolverConfig.h>
#include <suboptimal/solvers/linear/simplex.h>

#include <limits>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using namespace Eigen;
using namespace suboptimal;

TEST_CASE("Simplex failure mode - Degenerate cycling", "[simplex]") {
  auto problem = LinearProblem::maximizationProblem(Vector4d{{10, -57, -9, -24}});
  problem.addLessThanConstraint(Vector4d{{0.5, -5.5, -2.5, 9}}, 0);
  problem.addLessThanConstraint(Vector4d{{0.5, -1.5, -0.5, 1}}, 0);
  problem.addLessThanConstraint(Vector4d{{1, 1, 1, 1}}, 1);
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  VectorXd solution;
  double objective_value;

  SECTION("Max iterations exceeded") {
    auto status =
        solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = SimplexPivotRule::kDantzig});
    REQUIRE(status == SolverExitStatus::kMaxIterationsExceeded);
  }

  SECTION("Timeout") {
    auto status = solveSimplex(problem, solution, objective_value,
                               {.verbose = true,
                                .max_iterations = std::numeric_limits<int>::max(),
                                .timeout = std::chrono::duration<double, std::milli>{100},
                                .pivot_rule = SimplexPivotRule::kDantzig});
    REQUIRE(status == SolverExitStatus::kTimeout);
  }
}

TEST_CASE("Simplex falure mode - Unbounded problem", "[simplex]") {
  const auto pivot_rule =
      GENERATE(SimplexPivotRule::kLexicographic, SimplexPivotRule::kDantzig, SimplexPivotRule::kBland);

  auto problem = LinearProblem::maximizationProblem(Vector3d{{0, 2, 1}});
  problem.addLessThanConstraint(Vector3d{{1, -1, 1}}, 5);
  problem.addLessThanConstraint(Vector3d{{-2, 1, 0}}, 3);
  problem.addLessThanConstraint(Vector3d{{0, 1, -2}}, 5);
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  VectorXd solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});

  REQUIRE(status == SolverExitStatus::kUnbounded);
}

TEST_CASE("Simplex failure mode - Infeasible problem", "[simplex]") {
  const auto pivot_rule =
      GENERATE(SimplexPivotRule::kLexicographic, SimplexPivotRule::kDantzig, SimplexPivotRule::kBland);

  auto problem = LinearProblem::maximizationProblem(Vector3d{{1, -1, 1}});
  problem.addLessThanConstraint(Vector3d{{2, -1, -2}}, 4);
  problem.addGreaterThanConstraint(Vector3d{{-2, 3, 1}}, 5);
  problem.addGreaterThanConstraint(Vector3d{{1, -1, -1}}, 1);
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 2);

  VectorXd solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});

  REQUIRE(status == SolverExitStatus::kInfeasible);
}
