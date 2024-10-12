#include "suboptimal/solver/simplex.h"

#include <Eigen/Core>
#include <algorithm>
#include <stdexcept>
#include <vector>

using namespace suboptimal;
using namespace std;
using namespace Eigen;

SimplexProblem::SimplexProblem() : num_decision_vars(0), num_constraints(0), tableau(0, 0) {}

SimplexProblem& SimplexProblem::maximize(const VectorXd& objective_coeffs) {
  if (objective_coeffs.size() < 1) throw invalid_argument("Objective function must have at least one coefficient");
  if (num_decision_vars > 0) throw logic_error("Objective function has already been initialized");
  for (Index i = 0; i < objective_coeffs.size(); i++) {
    if (objective_coeffs(i) == 0) throw invalid_argument("Objective function coefficients must be non-zero");
  }

  c = objective_coeffs;
  A.resize(0, objective_coeffs.size());
  num_decision_vars = objective_coeffs.size();
  return *this;
}

SimplexProblem& SimplexProblem::constrainedBy(const VectorXd& constraint_coeffs, const double rhs) {
  if (constraint_coeffs.size() != num_decision_vars)
    throw invalid_argument("Constraint coefficients must have same dimension as decision variables");
  if (rhs < 0) throw invalid_argument("RHS of constraint must be non-negative");

  A.conservativeResize(++num_constraints, NoChange);
  b.conservativeResize(num_constraints);
  A.row(num_constraints - 1) = constraint_coeffs.transpose();
  b(num_constraints - 1) = rhs;
  return *this;
}

double SimplexProblem::solve(VectorXd& solution) {
  initTableau();
  Index pivot_row, pivot_col;
  while (true) {
    // Find pivot position, if valid position cannot be found, problem is unbounded
    if (!findPivotPosition(pivot_row, pivot_col)) {
      throw runtime_error("Failed to solve, problem is unbounded");
    }
    // Check end condition
    if (tableau.row(num_constraints)(pivot_col) >= 0) break;
    pivot(pivot_row, pivot_col);
  }

  solution.resize(num_decision_vars);
  for (Index i = 0; i < num_decision_vars; i++) {
    if (const Index row = getBasicRow(i); row >= 0)
      solution(i) = tableau(row, num_decision_vars + num_constraints + 1);
    else
      solution(i) = 0;
  }

  return tableau(tableau.rows() - 1, tableau.cols() - 1);
}

void SimplexProblem::initTableau() {
  tableau.resize(num_constraints + 1, num_decision_vars + num_constraints + 2);
  tableau.block(0, 0, num_constraints, num_decision_vars) = A;
  tableau.block(0, num_decision_vars, num_constraints + 1, num_constraints + 1) =
      MatrixXd::Identity(num_constraints + 1, num_constraints + 1);
  tableau.block(num_constraints, 0, 1, num_decision_vars) = -c.transpose();
  tableau.topRows(num_constraints).rightCols(1) = b;
}

bool SimplexProblem::findPivotPosition(Index& pivot_row, Index& pivot_col) {
  // Find column of most-negative objective function coefficient
  tableau.row(num_constraints).minCoeff(&pivot_col);

  const auto rhs = tableau.topRows(num_constraints).col(tableau.cols() - 1);
  double min_ratio = -1;
  vector<Index> valid_rows{};
  for (Index i = 0; i < num_constraints; i++) {
    if (const double ratio = rhs(i) / tableau(i, pivot_col); tableau(i, pivot_col) > 0 && ratio >= 0) {
      if (min_ratio == -1 || ratio < min_ratio) {
        min_ratio = ratio;
        valid_rows.clear();
        valid_rows.push_back(i);
      } else if (ratio == min_ratio) {
        valid_rows.push_back(i);
      }
    }
  }
  // No valid pivot, problem is unbounded
  if (valid_rows.empty()) return false;
  // Single pivot candidate, return it
  if (valid_rows.size() == 1) pivot_row = valid_rows[0];
  // Multiple pivot candidates, use lexicographic rule
  else {
    pivot_row = valid_rows[0];
    for (size_t i = 1; i < valid_rows.size(); i++) {
      const auto row_candidate = (tableau.row(valid_rows[i]) / tableau(valid_rows[i], pivot_col)).eval();
      const auto current_best_row = (tableau.row(pivot_row) / tableau(pivot_row, pivot_col)).eval();
      if (lexicographical_compare(row_candidate.data(), row_candidate.data() + row_candidate.size(),
                                  current_best_row.data(), current_best_row.data() + current_best_row.size())) {
        pivot_row = valid_rows[i];
      }
    }
  }
  return true;
}

void SimplexProblem::pivot(const Index pivot_row, const Index pivot_col) {
  const double pivot_element = tableau(pivot_row, pivot_col);
  auto pivot_row_vec = tableau.row(pivot_row);
  auto pivot_col_vec = tableau.col(pivot_col);

  pivot_row_vec *= 1 / pivot_element;
  for (Index i = 0; i < tableau.rows(); i++) {
    if (i == pivot_row) continue;
    tableau.row(i) += -pivot_col_vec(i) * pivot_row_vec;
  }
}

Index SimplexProblem::getBasicRow(const Index col) {
  auto col_vec = tableau.col(col);
  Index row = -1;
  for (Index i = 0; i < col_vec.size(); i++) {
    if (col_vec(i) == 1) {
      if (row >= 0) return -1;
      row = i;
    } else if (col_vec(i) != 0)
      return -1;
  }
  return row;
}

const MatrixXd& SimplexProblem::getTableau() { return tableau; }
