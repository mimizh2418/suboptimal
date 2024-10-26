#include "suboptimal/LinearProblem.h"

#include <Eigen/Core>
#include <format>
#include <gsl/narrow>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/expression_util.h"

using namespace suboptimal;
using namespace std;
using namespace Eigen;

LinearProblem::LinearProblem(const VectorXd& objective_coeffs)
    : objective_coeffs(objective_coeffs), num_decision_vars(objective_coeffs.size()) {
  if (objective_coeffs.size() < 1) {
    throw invalid_argument("Objective function must have at least one coefficient");
  }
  if ((objective_coeffs.array() == 0).any()) {
    throw invalid_argument("Objective function coefficients must be non-zero");
  }
}

LinearProblem LinearProblem::maximizationProblem(const VectorXd& objective_coeffs) {
  return LinearProblem(objective_coeffs);
}

void LinearProblem::addLessThanConstraint(const VectorXd& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, -1);
}

void LinearProblem::addGreaterThanConstraint(const VectorXd& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, 1);
}

void LinearProblem::addEqualityConstraint(const VectorXd& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, 0);
}

void LinearProblem::addConstraintImpl(const VectorXd& constraint_coeffs, const double rhs, const int constraint_type) {
  if (constraint_coeffs.size() != num_decision_vars) {
    throw invalid_argument("Constraint coefficients must have same dimension as decision variables");
  }

  VectorXd new_constraint(constraint_coeffs.size() + 1);
  new_constraint.head(num_decision_vars) = constraint_coeffs;
  new_constraint(num_decision_vars) = rhs;
  if (constraint_type == 0) {
    equality_constraints.push_back(new_constraint);
  } else if (constraint_type < 0) {
    less_than_constraints.push_back(new_constraint);
  } else if (constraint_type > 0) {
    greater_than_constraints.push_back(new_constraint);
  }

  num_constraints++;
}

void LinearProblem::buildConstraints(MatrixXd& constraint_matrix, VectorXd& constraint_rhs) const {
  const Index num_slack_vars = numLessThanConstraints() + numGreaterThanConstraints();
  const Index num_artificial_vars = numEqualityConstraints() + numGreaterThanConstraints();
  constraint_matrix = MatrixXd::Zero(num_constraints, num_decision_vars + num_slack_vars + num_artificial_vars);
  constraint_rhs = VectorXd::Zero(num_constraints);

  Index current_row_index = 0;
  Index slack_var_index = num_decision_vars;
  Index artificial_var_index = num_decision_vars + num_slack_vars;
  for (size_t i = 0; i < less_than_constraints.size(); i++, current_row_index++, slack_var_index++) {
    auto current_row = constraint_matrix.row(current_row_index);
    constraint_rhs(current_row_index) = less_than_constraints[i](num_decision_vars);
    current_row.head(num_decision_vars) = less_than_constraints[i].head(num_decision_vars);
    current_row(slack_var_index) = 1;
  }
  for (size_t i = 0; i < equality_constraints.size(); i++, current_row_index++, artificial_var_index++) {
    auto current_row = constraint_matrix.row(current_row_index);
    constraint_rhs(current_row_index) = equality_constraints[i](num_decision_vars);
    current_row.head(num_decision_vars) = equality_constraints[i].head(num_decision_vars);
    current_row(artificial_var_index) = 1;
  }
  for (size_t i = 0; i < greater_than_constraints.size();
       i++, current_row_index++, slack_var_index++, artificial_var_index++) {
    auto current_row = constraint_matrix.row(current_row_index);
    constraint_rhs(current_row_index) = greater_than_constraints[i](num_decision_vars);
    current_row.head(num_decision_vars) = greater_than_constraints[i].head(num_decision_vars);
    current_row(slack_var_index) = -1;
    current_row(artificial_var_index) = 1;
  }
}

Index LinearProblem::numEqualityConstraints() const {
  return gsl::narrow<Index>(equality_constraints.size());
}

Index LinearProblem::numLessThanConstraints() const {
  return gsl::narrow<Index>(less_than_constraints.size());
}

Index LinearProblem::numGreaterThanConstraints() const {
  return gsl::narrow<Index>(greater_than_constraints.size());
}

std::string LinearProblem::objectiveFunctionString() const {
  return expressionFromCoeffs(objective_coeffs, "x");
}

vector<string> LinearProblem::constraintStrings() const {
  vector<string> ret(num_constraints);
  size_t current_constraint_index = 0;
  for (size_t i = 0; i < equality_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] = expressionFromCoeffs(equality_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += format(" = {}", equality_constraints[i](num_decision_vars));
  }
  for (size_t i = 0; i < less_than_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] = expressionFromCoeffs(less_than_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += format(" ≤ {}", less_than_constraints[i](num_decision_vars));
  }
  for (size_t i = 0; i < greater_than_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] =
        expressionFromCoeffs(greater_than_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += format(" ≥ {}", greater_than_constraints[i](num_decision_vars));
  }
  return ret;
}
