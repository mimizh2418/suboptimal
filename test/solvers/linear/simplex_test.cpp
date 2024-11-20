// Copyright (c) 2024 Alvin Zhang.

#define CATCH_CONFIG_FAST_COMPILE

#include <suboptimal/LinearProblem.h>
#include <suboptimal/solvers/SolverExitStatus.h>
#include <suboptimal/solvers/linear/SimplexPivotRule.h>
#include <suboptimal/solvers/linear/SimplexSolverConfig.h>
#include <suboptimal/solvers/linear/simplex.h>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Eigen;
using namespace suboptimal;

TEST_CASE("Simplex - Basic 1-phase maximization problem", "[simplex]") {
  constexpr double expected_objective = 400;
  const Vector2d expected_solution{{4, 8}};

  const auto pivot_rule =
      GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Bland, SimplexPivotRule::Dantzig);

  auto problem = LinearProblem::maximizationProblem(Vector2d{{40, 30}});
  problem.addLessThanConstraint(Vector2d{{1, 1}}, 12);
  problem.addLessThanConstraint(Vector2d{{2, 1}}, 16);
  REQUIRE(problem.numSlackVars() == 2);
  REQUIRE(problem.numArtificialVars() == 0);

  Vector2d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});
  // Verify solution
  REQUIRE(status == SolverExitStatus::Success);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == expected_solution.size());
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}

TEST_CASE("Simplex - Basic 1-phase minimization problem", "[simplex]") {
  constexpr double expected_objective = -20;
  const Vector3d expected_solution{{0, 0, 5}};

  const auto pivot_rule =
      GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Bland, SimplexPivotRule::Dantzig);

  auto problem = LinearProblem::minimizationProblem(Vector3d{{-2, -3, -4}});
  problem.addLessThanConstraint(Vector3d{{3, 2, 1}}, 10);
  problem.addLessThanConstraint(Vector3d{{2, 5, 3}}, 15);
  REQUIRE(problem.numSlackVars() == 2);
  REQUIRE(problem.numArtificialVars() == 0);

  Vector3d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});
  // Verify solution
  REQUIRE(status == SolverExitStatus::Success);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == expected_solution.size());
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}

TEST_CASE("Simplex - Degenerate 1-phase problem", "[simplex]") {
  constexpr double expected_objective = 0.5;
  const Vector4d expected_solution{{0.5, 0, 0.5, 0}};

  const auto pivot_rule = GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Bland);

  auto problem = LinearProblem::maximizationProblem(Vector4d{{10, -57, -9, -24}});
  problem.addLessThanConstraint(Vector4d{{0.5, -5.5, -2.5, 9}}, 0);
  problem.addLessThanConstraint(Vector4d{{0.5, -1.5, -0.5, 1}}, 0);
  problem.addLessThanConstraint(Vector4d{{1, 1, 1, 1}}, 1);
  REQUIRE(problem.numSlackVars() == 3);
  REQUIRE(problem.numArtificialVars() == 0);

  Vector4d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});
  // Verify solution
  REQUIRE(status == SolverExitStatus::Success);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == expected_solution.size());
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}

TEST_CASE("Simplex - Basic 2-phase problem", "[simplex]") {
  constexpr double expected_objective = -130.0 / 7.0;
  const Vector3d expected_solution{{15.0 / 7.0, 0.0, 25.0 / 7.0}};

  const auto pivot_rule =
      GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Bland, SimplexPivotRule::Dantzig);

  auto problem = LinearProblem::minimizationProblem(Vector3d{{-2, -3, -4}});
  problem.addEqualityConstraint(Vector3d{{3, 2, 1}}, 10);
  problem.addEqualityConstraint(Vector3d{{2, 5, 3}}, 15);
  REQUIRE(problem.numSlackVars() == 0);
  REQUIRE(problem.numArtificialVars() == 2);

  Vector3d solution;
  double objective_value;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});
  // Verify solution
  REQUIRE(status == SolverExitStatus::Success);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == expected_solution.size());
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}

TEST_CASE("Simplex - Degenerate 2-phase problem", "[simplex]") {
  constexpr double expected_objective = 11;
  const Vector4d expected_solution{{4, 0, 1, 2}};

  const auto pivot_rule = GENERATE(SimplexPivotRule::Lexicographic, SimplexPivotRule::Bland);

  auto problem = LinearProblem::minimizationProblem(Vector4d{{2, 6, 1, 1}});
  problem.addEqualityConstraint(Vector4d{{1, 2, 0, 1}}, 6);
  problem.addEqualityConstraint(Vector4d{{1, 2, 1, 1}}, 7);
  problem.addEqualityConstraint(Vector4d{{1, 3, -1, 2}}, 7);
  problem.addEqualityConstraint(Vector4d{{1, 1, 1, 0}}, 5);
  REQUIRE(problem.numSlackVars() == 0);
  REQUIRE(problem.numArtificialVars() == 4);

  double objective_value;
  Vector4d solution;
  auto status = solveSimplex(problem, solution, objective_value, {.verbose = true, .pivot_rule = pivot_rule});

  // Verify solution
  REQUIRE(status == SolverExitStatus::Success);
  CHECK_THAT(objective_value, Catch::Matchers::WithinAbs(expected_objective, 1e-9));
  CHECK(solution.size() == expected_solution.size());
  for (Index i = 0; i < solution.size(); i++) {
    CHECK_THAT(solution(i), Catch::Matchers::WithinAbs(expected_solution(i), 1e-9));
  }
}
