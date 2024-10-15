#include "suboptimal/LinearProblem.h"

#include <Eigen/Core>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

using namespace suboptimal;
using namespace std;
using namespace Eigen;

LinearProblem::LinearProblem(const VectorXd& objective_coeffs)
    : A(0, objective_coeffs.size()),
      b(0),
      c(objective_coeffs),
      num_decision_vars(objective_coeffs.size()),
      num_constraints(0) {
  if (objective_coeffs.size() < 1) throw invalid_argument("Objective function must have at least one coefficient");
  for (Index i = 0; i < objective_coeffs.size(); i++) {
    if (objective_coeffs(i) == 0) throw invalid_argument("Objective function coefficients must be non-zero");
  }
}

LinearProblem LinearProblem::maximize(const VectorXd& objective_coeffs) { return LinearProblem(objective_coeffs); }

LinearProblem& LinearProblem::constrainedBy(const VectorXd& constraint_coeffs, const double rhs) {
  if (constraint_coeffs.size() != num_decision_vars)
    throw invalid_argument("Constraint coefficients must have same dimension as decision variables");
  if (rhs < 0) throw invalid_argument("RHS of constraint must be non-negative");

  A.conservativeResize(++num_constraints, NoChange);
  b.conservativeResize(num_constraints);
  A.row(num_constraints - 1) = constraint_coeffs.transpose();
  b(num_constraints - 1) = rhs;
  return *this;
}

const VectorXd& LinearProblem::getObjectiveCoeffs() const { return c; }

const MatrixXd& LinearProblem::getConstraintMatrix() const { return A; }

const VectorXd& LinearProblem::getConstraintRHS() const { return b; }

std::string LinearProblem::objectiveFunctionString() const {
  string ret = format("{:.3f}x_1", c(0));
  for (Index i = 1; i < c.size(); i++) {
    ret = format("{}{}{:.3f}x_{}", ret, c(i) > 0 ? " + " : " - ", abs(c(i)), i + 1);
  }
  return ret;
}

vector<string> LinearProblem::constraintStrings() const {
  vector<string> ret(num_constraints);
  for (Index i = 0; i < num_constraints; i++) {
    ret[i] = format("{:.3f}x_1", A(i, 0));
    for (Index j = 1; j < num_decision_vars; j++) {
      if (A(i, j) == 0) continue;
      ret[i] = format("{}{}{:.3f}x_{}", ret[i], A(i, j) > 0 ? " + " : " - ", abs(A(i, j)), j + 1);
    }
    ret[i] += format(" ≤ {:.3f}", b(i));
  }
  return ret;
}
