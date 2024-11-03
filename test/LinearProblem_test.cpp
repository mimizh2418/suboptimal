// Copyright (c) 2024 Alvin Zhang.

#include <suboptimal/LinearProblem.h>

#include <Eigen/Core>
#include <catch2/catch_test_macros.hpp>

using namespace suboptimal;
using namespace Eigen;

TEST_CASE("LinearProblem - Problem initialization") {
  auto problem = LinearProblem::maximizationProblem(Vector2d{{3, 2}});
  problem.addLessThanConstraint(Vector2d{{2, 1}}, 50);
  problem.addGreaterThanConstraint(Vector2d{{1, 3}}, 15);
  problem.addEqualityConstraint(Vector2d{{5, 6}}, 60);

  CHECK(problem.numConstraints() == 3);
  CHECK(problem.numDecisionVars() == 2);
  CHECK(problem.numSlackVars() == 2);
  CHECK(problem.numArtificialVars() == 2);

  const Matrix<double, 3, 6> expected_mat{{2, 1, 1, 0, 0, 0},    //
                                          {5, 6, 0, 0, 1, 0},    //
                                          {1, 3, 0, -1, 0, 1}};  //
  const Vector3d expected_rhs{{50, 60, 15}};

  Matrix<double, 3, 6> mat;
  Vector3d rhs;
  problem.buildConstraints(mat, rhs);

  REQUIRE(mat.rows() == expected_mat.rows());
  REQUIRE(mat.cols() == expected_mat.cols());
  REQUIRE(rhs.size() == expected_rhs.size());

  CHECK(mat == expected_mat);
  CHECK(rhs == expected_rhs);
}
