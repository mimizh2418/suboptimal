#include "suboptimal/solvers/linear/SimplexPivotRule.h"
#include "suboptimal/solvers/linear/SimplexSolverConfig.h"
#include "suboptimal/solvers/linear/simplex.h"

using namespace Eigen;

void solveBasicProblem() {
  auto problem = suboptimal::LinearProblem::maximizationProblem(VectorXd{{40, 30}});
  problem.addLessThanConstraint(VectorXd{{1, 1}}, 12);
  problem.addLessThanConstraint(VectorXd{{2, 1}}, 16);

  suboptimal::SimplexSolverConfig solver_config;
  solver_config.verbose = true;

  VectorXd solution;
  double objective_value;
  solveSimplex(problem, solution, objective_value, solver_config);
}

void solveCyclingProblem(const suboptimal::SimplexPivotRule pivot_rule) {
  auto problem = suboptimal::LinearProblem::maximizationProblem(VectorXd{{10, -57, -9, -24}});
  problem.addLessThanConstraint(VectorXd{{0.5, -5.5, -2.5, 9}}, 0);
  problem.addLessThanConstraint(VectorXd{{0.5, -1.5, -0.5, 1}}, 0);
  problem.addLessThanConstraint(VectorXd{{1, 1, 1, 1}}, 1);

  suboptimal::SimplexSolverConfig solver_config;
  solver_config.pivot_rule = pivot_rule;
  solver_config.verbose = true;

  VectorXd solution;
  double objective_value;
  solveSimplex(problem, solution, objective_value, solver_config);
}

void solveDegenerate2PhaseProblem() {
  auto problem = suboptimal::LinearProblem::maximizationProblem(VectorXd{{-2, -6, -1, -1}});
  problem.addEqualityConstraint(VectorXd{{1, 2, 0, 1}}, 6);
  problem.addEqualityConstraint(VectorXd{{1, 2, 1, 1}}, 7);
  problem.addEqualityConstraint(VectorXd{{1, 3, -1, 2}}, 7);
  problem.addEqualityConstraint(VectorXd{{1, 1, 1, 0}}, 5);

  suboptimal::SimplexSolverConfig solver_config;
  solver_config.verbose = true;
  solver_config.pivot_rule = suboptimal::SimplexPivotRule::kLexicographic;

  VectorXd solution;
  double objective_value;
  solveSimplex(problem, solution, objective_value, solver_config);
}

int main() {
  solveBasicProblem();                                                // Very simple problem
  solveCyclingProblem(suboptimal::SimplexPivotRule::kLexicographic);  // Cycling problem with lexicographic pivot rule
  solveCyclingProblem(suboptimal::SimplexPivotRule::kBland);          // Cycling problem with Bland's pivot rule
  solveCyclingProblem(suboptimal::SimplexPivotRule::kDantzig);  // Cycling problem with Dantzig's pivot rule (fails)
  solveDegenerate2PhaseProblem();                               // Degenerate problem with 2-phase simplex
}
