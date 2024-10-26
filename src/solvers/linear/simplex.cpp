#include "suboptimal/solvers/linear/simplex.h"

#include <Eigen/Core>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "suboptimal/LinearProblem.h"
#include "suboptimal/solvers/SolverExitStatus.h"
#include "suboptimal/solvers/linear/SimplexPivotRule.h"
#include "suboptimal/solvers/linear/SimplexSolverConfig.h"
#include "util/SolverProfiler.h"
#include "util/comparison_util.h"

using namespace suboptimal;
using namespace Eigen;
using namespace std;

int findPivotPosition(const MatrixXd& tableau, const VectorX<Index>& basic_vars, const SimplexPivotRule pivot_rule,
                      Index& pivot_row, Index& pivot_col) {
  pivot_col = -1;
  pivot_row = -1;

  // Entering variable selection
  const auto objective_row = tableau.row(tableau.rows() - 1).head(tableau.cols() - 1);
  if (pivot_rule == SimplexPivotRule::kBland) {
    // Bland's rule: select the index of the first negative coefficient in the objective row
    for (Index i = 0; i < objective_row.size(); i++) {
      if (approxGEQ<double>(objective_row(i), 0)) {
        continue;
      }
      pivot_col = i;
      break;
    }
  } else if (pivot_rule == SimplexPivotRule::kLexicographic || pivot_rule == SimplexPivotRule::kDantzig) {
    // Dantzig's or lexicographic rule: select the index of the most negative coefficient in the objective row
    Index smallest_idx;
    if (const double smallest_coeff = objective_row.minCoeff(&smallest_idx); approxLT<double>(smallest_coeff, 0)) {
      pivot_col = smallest_idx;
    }
  }

  // If no negative coefficients are found, the solution is optimal
  if (pivot_col == -1) {
    return 1;
  }

  // Leaving variable selection
  double min_ratio = numeric_limits<double>::infinity();
  const auto rhs_col = tableau.col(tableau.cols() - 1).head(tableau.rows() - 1);
  for (Index i = 0; i < rhs_col.size(); i++) {
    const double ratio = rhs_col(i) / tableau(i, pivot_col);
    // Pivot element cannot be 0 and ratio cannot be negative
    if (approxLEQ<double>(tableau(i, pivot_col), 0) || approxLT<double>(ratio, 0)) {
      continue;
    }

    // Minimum ratio test
    if (approxLT<double>(ratio, min_ratio)) {
      min_ratio = ratio;
      pivot_row = i;
    } else if (isApprox<double>(ratio, min_ratio) && pivot_rule == SimplexPivotRule::kLexicographic) {
      // Lexicographic rule: select the variable with the lexicographically smallest row
      const RowVectorXd row_candidate = tableau.row(i) / tableau(i, pivot_col);
      const RowVectorXd current_best_row = tableau.row(pivot_row) / tableau(pivot_row, pivot_col);
      if (lexicographical_compare(row_candidate.data(), row_candidate.data() + row_candidate.size(),
                                  current_best_row.data(), current_best_row.data() + current_best_row.size())) {
        pivot_row = i;
      }
    } else if (isApprox<double>(ratio, min_ratio) && pivot_rule == SimplexPivotRule::kBland) {
      // Bland's rule: select the basic variable with the smallest index
      if (basic_vars(i) < basic_vars(pivot_row)) {
        pivot_row = i;
      }
    }
  }

  // If no valid pivot is found, the problem is unbounded
  if (pivot_row == -1) {
    return -1;
  }

  // Valid pivot found
  return 0;
}

void pivot(MatrixXd& tableau, VectorX<Index>& basic_vars, const Index pivot_row, const Index pivot_col) {
  const double pivot_element = tableau(pivot_row, pivot_col);
  tableau.row(pivot_row) /= pivot_element;
  tableau(pivot_row, pivot_col) = 1;  // Account for floating point errors
  for (Index i = 0; i < tableau.rows(); i++) {
    if (i == pivot_row || isApprox<double>(tableau(i, pivot_col), 0)) {
      continue;
    }
    tableau.row(i) -= tableau(i, pivot_col) * tableau.row(pivot_row);
    tableau(i, pivot_col) = 0;  // Account for floating point errors
  }
  // Update basic variables
  basic_vars(pivot_row) = pivot_col;
}

VectorX<Index> findBasicVars(const MatrixXd& tableau) {
  VectorX<Index> basic_vars = VectorX<Index>::Zero(tableau.rows() - 1);
  for (Index i = 0; i < tableau.cols(); i++) {
    const auto col = tableau.col(i);
    Index max_index;
    if (isApprox<double>(col.lpNorm<1>(), 1) && isApprox<double>(col.maxCoeff(&max_index), 1)) {
      basic_vars(max_index) = i;
    }
  }
  return basic_vars;
}

SolverExitStatus solveTableau(MatrixXd& tableau, VectorX<Index>& basic_vars, SolverProfiler& profiler,
                              const SimplexSolverConfig& config) {
  int num_iterations = 0;
  auto exit_status = SolverExitStatus::kSuccess;

  while (true) {
    profiler.startIteration();
    // Find pivot position
    Index pivot_row, pivot_col;
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

    // Check for maximum iterations
    if (++num_iterations >= config.max_iterations) {
      exit_status = SolverExitStatus::kMaxIterationsExceeded;
      break;
    }

    profiler.endIteration();
  }
  profiler.endIteration();

  return exit_status;
}

namespace suboptimal {
SolverExitStatus solveSimplex(const LinearProblem& problem, VectorXd& solution, double& objective_value,
                              const SimplexSolverConfig& config) {
  MatrixXd constraint_matrix;
  VectorXd constraint_rhs;
  problem.buildConstraints(constraint_matrix, constraint_rhs);
  if (problem.numConstraints() == 0) {
    throw invalid_argument("Problem must have at least one constraint");
  }

  if (config.verbose) {
    cout << "Solving linear problem: " << endl;
    cout << "Maximize: " << problem.objectiveFunctionString() << endl;
    cout << "Subject to: " << endl;
    for (const auto constraint_strings = problem.constraintStrings();
         const auto& constraint_string : constraint_strings) {
      cout << "  " << constraint_string << endl;
    }
    cout << "Using pivot rule: " << toString(config.pivot_rule) << endl << endl;
  }

  // Initialize tableau
  MatrixXd tableau = MatrixXd::Zero(problem.numConstraints() + 1, constraint_matrix.cols() + 1);
  tableau.topLeftCorner(problem.numConstraints(), constraint_matrix.cols()) = constraint_matrix;
  tableau.topRightCorner(problem.numConstraints(), 1) = constraint_rhs;
  tableau.bottomLeftCorner(1, problem.numDecisionVars()) = -problem.getObjectiveCoeffs().transpose();

  if (!problem.hasInitialBFS()) {
    if (config.verbose) {
      cout << "No trivial BFS found, solving auxiliary LP" << endl;
    }

    // Set up auxiliary LP
    const Index num_artificial_vars = problem.numEqualityConstraints() + problem.numGreaterThanConstraints();

    // Construct auxiliary objective function
    RowVectorXd auxiliary_objective = RowVectorXd::Zero(tableau.cols());
    auxiliary_objective.segment(problem.numDecisionVars() + problem.numSlackVars(), num_artificial_vars) =
        RowVectorXd::Ones(num_artificial_vars);

    // Subtract rows with artificial variables from the auxiliary objective to make it a valid objective function
    const auto artificial_rows = tableau.middleRows(problem.numLessThanConstraints(), num_artificial_vars);
    RowVectorXd artificial_row_sum = RowVectorXd::Zero(tableau.cols());
    for (Index i = 0; i < artificial_rows.rows(); i++) {
      artificial_row_sum += artificial_rows.row(i);
    }
    auxiliary_objective -= artificial_row_sum;

    // Replace the original objective function with the auxiliary objective
    tableau.row(tableau.rows() - 1) = auxiliary_objective;

    // Find basic variables
    VectorX<Index> basic_vars = findBasicVars(tableau);

    // Perform simplex iterations to find initial BFS
    SolverProfiler aux_profiler{};
    const auto aux_exit = solveTableau(tableau, basic_vars, aux_profiler, config);

    if (config.verbose) {
      const auto total_time = aux_profiler.getAvgIterationTime() * aux_profiler.numIterations();
      cout << format("Auxiliary LP solve time: {:.3f} ms ({} iterations; {:.3f} ms average)", total_time.count(),
                     aux_profiler.numIterations(), aux_profiler.getAvgIterationTime().count())
           << endl
           << endl;
    }

    if (aux_exit == SolverExitStatus::kMaxIterationsExceeded) {
      if (config.verbose) {
        cout << "Max iterations exceeded while solving auxiliary LP" << endl;
      }
      return aux_exit;
    }
    if (aux_exit == SolverExitStatus::kUnbounded ||
        !isApprox<double>(tableau(tableau.rows() - 1, tableau.cols() - 1), 0)) {
      cout << "The problem is infeasible" << endl;
      return SolverExitStatus::kInfeasible;
    }

    auto objective_row = tableau.row(tableau.rows() - 1);
    objective_row.head(problem.numDecisionVars()) = -problem.getObjectiveCoeffs().transpose();
    for (Index i = 0; i < basic_vars.size(); i++) {
      objective_row -= tableau.row(i) * objective_row(basic_vars(i));
    }
  }

  // Initialize basic variables
  VectorX<Index> basic_vars = findBasicVars(tableau);

  // Perform simplex iterations
  SolverProfiler profiler{};
  const auto exit_status = solveTableau(tableau, basic_vars, profiler, config);

  if (config.verbose) {
    const auto total_time = profiler.getAvgIterationTime() * profiler.numIterations();
    cout << format("Solve time: {:.3f} ms ({} iterations; {:.3f} ms average)", total_time.count(),
                   profiler.numIterations(), profiler.getAvgIterationTime().count())
         << endl
         << "Status: " << toString(exit_status) << endl;
  }

  if (exit_status == SolverExitStatus::kSuccess) {
    // Extract solution and objective value
    solution = VectorXd::Zero(problem.numDecisionVars());
    const VectorXd rhs = tableau.col(tableau.cols() - 1);
    for (Index i = 0; i < basic_vars.size(); i++) {
      if (const Index var_index = basic_vars(i); var_index < problem.numDecisionVars()) {
        solution(var_index) = rhs(i);
      }
    }
    objective_value = tableau(tableau.rows() - 1, tableau.cols() - 1);

    if (config.verbose) {
      cout << "Solution: " << endl;
      for (Index i = 0; i < problem.numDecisionVars(); i++) {
        cout << "  x_" << i + 1 << " = " << solution(i) << endl;
      }
      cout << "Objective value: " << objective_value << endl;
    }
  }

  if (config.verbose) {
    cout << endl;
  }

  return exit_status;
}
}  // namespace suboptimal
