// Copyright (c) 2024 Alvin Zhang.

#include <suboptimal/LinearProblem.h>
#include <suboptimal/solvers/SolverExitStatus.h>
#include <suboptimal/solvers/linear/SimplexPivotRule.h>
#include <suboptimal/solvers/linear/SimplexSolverConfig.h>
#include <suboptimal/solvers/linear/simplex.h>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Eigen;
using namespace suboptimal;

TEST_CASE("Simplex - Basic 1-phase problem", "[simplex]") {
  constexpr double expected_objective = 400;
  const Vector2d expected_solution{{4, 8}};

  auto problem = LinearProblem::maximizationProblem(Vector2d{{40, 30}});
  problem.addLessThanConstraint(Vector2d{{1, 1}}, 12);
  problem.addLessThanConstraint(Vector2d{{2, 1}}, 16);
  REQUIRE(problem.numSlackVars() == 2);
  REQUIRE(problem.numArtificialVars() == 0);

  VectorXd solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true});
  // Verify solution
  REQUIRE(status == SolverExitStatus::kSuccess);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == 2);
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}

TEST_CASE("Simplex - Degenerate 1-phase problem", "[simplex]") {
  constexpr double expected_objective = 0.5;
  const Vector4d expected_solution{{0.5, 0, 0.5, 0}};

  auto problem = LinearProblem::maximizationProblem(Vector4d{{10, -57, -9, -24}});
  problem.addLessThanConstraint(Vector4d{{0.5, -5.5, -2.5, 9}}, 0);
  problem.addLessThanConstraint(Vector4d{{0.5, -1.5, -0.5, 1}}, 0);
  problem.addLessThanConstraint(Vector4d{{1, 1, 1, 1}}, 1);
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  // Solve with lexicographic pivot rule
  VectorXd solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value,
                             {.verbose = true, .pivot_rule = SimplexPivotRule::kLexicographic});
  // Verify solution
  REQUIRE(status == SolverExitStatus::kSuccess);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == 4);
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }

  // Solve with Bland's pivot rule
  status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = SimplexPivotRule::kBland});
  // Verify solution
  REQUIRE(status == SolverExitStatus::kSuccess);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == 4);
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }

  // Solve with Dantzig's pivot rule (should cycle)
  status =
      solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = SimplexPivotRule::kDantzig});
  REQUIRE(status == SolverExitStatus::kMaxIterationsExceeded);
}

TEST_CASE("Simplex - Degenerate 2-phase problem", "[simplex]") {
  constexpr double expected_objective = -11;
  const Vector4d expected_solution{{4, 0, 1, 2}};

  auto problem = LinearProblem::maximizationProblem(Vector4d{{-2, -6, -1, -1}});
  problem.addEqualityConstraint(Vector4d{{1, 2, 0, 1}}, 6);
  problem.addEqualityConstraint(Vector4d{{1, 2, 1, 1}}, 7);
  problem.addEqualityConstraint(Vector4d{{1, 3, -1, 2}}, 7);
  problem.addEqualityConstraint(Vector4d{{1, 1, 1, 0}}, 5);
  REQUIRE(problem.numSlackVars() == 0);
  REQUIRE(problem.numArtificialVars() == 4);

  double objective_value;
  VectorXd solution;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true});

  // Verify solution
  REQUIRE(status == SolverExitStatus::kSuccess);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == 4);
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}
