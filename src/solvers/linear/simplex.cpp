#include "suboptimal/solvers/linear/simplex.h"

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/SolverExitStatus.h"
#include "suboptimal/solvers/linear/SimplexPivotRule.h"
#include "suboptimal/solvers/linear/SimplexSolverConfig.h"
#include "util/SolverProfiler.h"

using namespace suboptimal;
using namespace Eigen;
using namespace std;

int findPivotPosition(const MatrixXd& tableau, const span<Index> basic_vars, const SimplexPivotRule pivot_rule,
                      Index& pivot_row, Index& pivot_col) {
  pivot_col = -1;
  pivot_row = -1;

  // Entering variable selection
  const auto objective_row = tableau.row(tableau.rows() - 1).head(tableau.cols() - 1);
  if (pivot_rule == SimplexPivotRule::kBland) {
    // Bland's rule: select the index of the first negative coefficient in the objective row
    for (Index i = 0; i < objective_row.size(); i++) {
      if (objective_row(i) < 0) {
        pivot_col = i;
        break;
      }
    }
  } else if (pivot_rule == SimplexPivotRule::kLexicographic || pivot_rule == SimplexPivotRule::kDantzig) {
    // Dantzig's or lexicographic rule: select the index of the most negative coefficient in the objective row
    Index smallest_idx;
    if (const double smallest_coeff = objective_row.minCoeff(&smallest_idx); smallest_coeff < 0)
      pivot_col = smallest_idx;
  }

  // If no negative coefficients are found, the solution is optimal
  if (pivot_col == -1) return 1;

  // Leaving variable selection
  double min_ratio = numeric_limits<double>::infinity();
  const auto rhs_col = tableau.col(tableau.cols() - 1).head(tableau.rows() - 1);
  for (Index i = 0; i < rhs_col.size(); i++) {
    // Minimum ratio test, find the smallest, non-negative ratio with non-negative pivot element
    if (const double ratio = rhs_col(i) / tableau(i, pivot_col); tableau(i, pivot_col) > 0 && ratio >= 0) {
      if (ratio < min_ratio) {
        min_ratio = ratio;
        pivot_row = i;
      } else if (ratio == min_ratio && pivot_rule == SimplexPivotRule::kLexicographic) {
        // Lexicographic rule: select the variable with the lexicographically smallest row
        const auto row_candidate = (tableau.row(i) / tableau(i, pivot_col)).eval();
        const auto current_best_row = (tableau.row(pivot_row) / tableau(pivot_row, pivot_col)).eval();
        if (lexicographical_compare(row_candidate.data(), row_candidate.data() + row_candidate.size(),
                                    current_best_row.data(), current_best_row.data() + current_best_row.size())) {
          pivot_row = i;
        }
      } else if (ratio == min_ratio && pivot_rule == SimplexPivotRule::kBland) {
        // Bland's rule: select the basic variable with the smallest index
        if (basic_vars[i] < basic_vars[pivot_row]) {
          pivot_row = i;
        }
      }
    }
  }

  // If no valid pivot is found, the problem is unbounded
  if (pivot_row == -1) return -1;

  // Valid pivot found
  return 0;
}

void pivot(MatrixXd& tableau, span<Index> basic_vars, const Index pivot_row, const Index pivot_col) {
  const double pivot_element = tableau(pivot_row, pivot_col);
  tableau.row(pivot_row) /= pivot_element;
  tableau(pivot_row, pivot_col) = 1;  // Account for floating point errors
  for (Index i = 0; i < tableau.rows(); i++) {
    if (i == pivot_row) continue;
    tableau.row(i) -= tableau(i, pivot_col) * tableau.row(pivot_row);
    tableau(i, pivot_col) = 0;  // Account for floating point errors
  }

  // Update basic variables
  basic_vars[pivot_row] = pivot_col;
}

namespace suboptimal {
SolverExitStatus solveSimplex(const LinearProblem& problem, VectorXd& solution, double& objective_value,
                              const SimplexSolverConfig& config) {
  const Index num_decision_vars = problem.getObjectiveCoeffs().size();
  const Index num_constraints = problem.getConstraintMatrix().rows();
  if (num_constraints == 0) throw invalid_argument("Problem must have at least one constraint");

  SolverExitStatus exit_status;

  if (config.verbose) {
    cout << "Solving linear problem: " << endl;
    cout << "Maximize: " << problem.objectiveFunctionString() << endl;
    cout << "Subject to: " << endl;
    const auto constraint_strings = problem.constraintStrings();
    for (int i = 0; i < num_constraints; i++) {
      cout << "  " << constraint_strings[i] << endl;
    }
    cout << "Using pivot rule: " << toString(config.pivot_rule) << endl << endl;
  }

  // Initialize tableau
  MatrixXd tableau = MatrixXd::Zero(num_constraints + 1, num_decision_vars + num_constraints + 2);
  tableau.topLeftCorner(num_constraints, num_decision_vars) = problem.getConstraintMatrix();
  tableau.block(0, num_decision_vars, num_constraints + 1, num_constraints + 1) =
      MatrixXd::Identity(num_constraints + 1, num_constraints + 1);
  tableau.topRightCorner(num_constraints, 1) = problem.getConstraintRHS();
  tableau.bottomLeftCorner(1, num_decision_vars) = -problem.getObjectiveCoeffs().transpose();

  // Initialize basic variables
  vector<Index> basic_vars(num_constraints);
  for (Index i = 0; i < num_constraints; i++) {
    basic_vars[i] = num_decision_vars + i;
  }

  // Perform simplex iterations
  SolverProfiler profiler;
  Index pivot_row, pivot_col;
  int num_iterations = 0;
  while (true) {
    profiler.startIteration();
    // Find pivot position
    const int pivot_status = findPivotPosition(tableau, basic_vars, config.pivot_rule, pivot_row, pivot_col);
    if (pivot_status == -1) {
      exit_status = SolverExitStatus::kUnbounded;  // Could not find a valid pivot position, problem is unbounded
      break;
    }
    if (pivot_status == 1) {
      exit_status = SolverExitStatus::kSuccess;  // Optimal solution found
      break;
    }
    // Perform pivot operation
    pivot(tableau, basic_vars, pivot_row, pivot_col);

    profiler.endIteration();

    // Check for maximum iterations
    if (++num_iterations >= config.max_iterations) {
      exit_status = SolverExitStatus::kMaxIterationsExceeded;
      break;
    }
  }

  if (config.verbose) {
    const auto total_time = profiler.getAvgIterationTime() * profiler.getNumIterations();
    cout << format("Solve time: {:.3f} ms ({} iterations; {:.3f} ms average)", total_time.count(),
                   profiler.getNumIterations(), profiler.getAvgIterationTime().count())
         << endl
         << "Status: " << toString(exit_status) << endl;
  }

  if (exit_status == SolverExitStatus::kSuccess) {
    // Extract solution and objective value
    solution = VectorXd::Zero(num_decision_vars);
    auto rhs = tableau.col(tableau.cols() - 1);
    for (size_t i = 0; i < basic_vars.size(); i++) {
      if (const Index var_index = basic_vars[i]; var_index < num_decision_vars) {
        solution(var_index) = rhs(i);
      }
    }
    objective_value = tableau(tableau.rows() - 1, tableau.cols() - 1);

    if (config.verbose) {
      cout << "Solution: " << endl;
      for (Index i = 0; i < num_decision_vars; i++) {
        cout << "  x_" << i + 1 << " = " << solution(i) << endl;
      }
      cout << "Objective value: " << objective_value << endl << endl;
    }
  }

  return exit_status;
}
}  // namespace suboptimal