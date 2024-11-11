// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/LinearProblem.h"

#include <format>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <gsl/narrow>

#include "util/comparison_util.h"
#include "util/expression_util.h"

using namespace suboptimal;
using namespace Eigen;

LinearProblem::LinearProblem(const Ref<const VectorXd>& objective_coeffs, const bool is_minimization)
    : is_minimization(is_minimization),
      objective_coeffs((is_minimization ? -1 : 1) * objective_coeffs),
      num_decision_vars(objective_coeffs.size()) {
  Expects(objective_coeffs.size() > 0);
}

LinearProblem LinearProblem::maximizationProblem(const Ref<const VectorXd>& objective_coeffs) {
  return LinearProblem(objective_coeffs, false);
}

LinearProblem LinearProblem::minimizationProblem(const Ref<const VectorXd>& objective_coeffs) {
  return LinearProblem(objective_coeffs, true);
}

void LinearProblem::addLessThanConstraint(const Ref<const VectorXd>& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, -1);
}

void LinearProblem::addGreaterThanConstraint(const Ref<const VectorXd>& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, 1);
}

void LinearProblem::addEqualityConstraint(const Ref<const VectorXd>& constraint_coeffs, const double rhs) {
  addConstraintImpl(constraint_coeffs, rhs, 0);
}

void LinearProblem::addConstraintImpl(const Ref<const VectorXd>& constraint_coeffs, const double rhs,
                                      const int constraint_type) {
  Expects(constraint_coeffs.size() == num_decision_vars);

  VectorXd coeffs = constraint_coeffs;
  double new_rhs = rhs;
  int type = constraint_type;
  if (approxLT<double>(rhs, 0)) {
    coeffs = -constraint_coeffs;
    new_rhs = -rhs;
    type = -constraint_type;
  }

  VectorXd new_constraint(coeffs.size() + 1);
  new_constraint.head(num_decision_vars) = coeffs;
  new_constraint(num_decision_vars) = new_rhs;
  if (type == 0) {
    equality_constraints.push_back(new_constraint);
  } else if (type < 0) {
    less_than_constraints.push_back(new_constraint);
  } else if (type > 0) {
    greater_than_constraints.push_back(new_constraint);
  }

  num_constraints++;
}

void LinearProblem::buildConstraints(Ref<MatrixXd> constraint_matrix, Ref<VectorXd> constraint_rhs) const {
  Expects(constraint_matrix.rows() == num_constraints &&
          constraint_matrix.cols() == num_decision_vars + numSlackVars() + numArtificialVars());
  Expects(constraint_rhs.rows() == num_constraints);

  constraint_matrix = MatrixXd::Zero(num_constraints, num_decision_vars + numSlackVars() + numArtificialVars());
  constraint_rhs = VectorXd::Zero(num_constraints);

  Index current_row_index = 0;
  Index slack_var_index = num_decision_vars;
  Index artificial_var_index = num_decision_vars + numSlackVars();
  for (size_t i = 0; i < less_than_constraints.size(); i++, current_row_index++, slack_var_index++) {
    Ref<MatrixXd>::RowXpr current_row = constraint_matrix.row(current_row_index);
    constraint_rhs(current_row_index) = less_than_constraints[i](num_decision_vars);
    current_row.head(num_decision_vars) = less_than_constraints[i].head(num_decision_vars);
    current_row(slack_var_index) = 1;
  }
  for (size_t i = 0; i < equality_constraints.size(); i++, current_row_index++, artificial_var_index++) {
    Ref<MatrixXd>::RowXpr current_row = constraint_matrix.row(current_row_index);
    constraint_rhs(current_row_index) = equality_constraints[i](num_decision_vars);
    current_row.head(num_decision_vars) = equality_constraints[i].head(num_decision_vars);
    current_row(artificial_var_index) = 1;
  }
  for (size_t i = 0; i < greater_than_constraints.size();
       i++, current_row_index++, slack_var_index++, artificial_var_index++) {
    Ref<MatrixXd>::RowXpr current_row = constraint_matrix.row(current_row_index);
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
  return expressionFromCoeffs((is_minimization ? -1 : 1) * objective_coeffs, "x");
}

std::vector<std::string> LinearProblem::constraintStrings() const {
  std::vector<std::string> ret(num_constraints);
  size_t current_constraint_index = 0;
  for (size_t i = 0; i < equality_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] = expressionFromCoeffs(equality_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += std::format(" = {}", equality_constraints[i](num_decision_vars));
  }
  for (size_t i = 0; i < less_than_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] = expressionFromCoeffs(less_than_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += std::format(" ≤ {}", less_than_constraints[i](num_decision_vars));
  }
  for (size_t i = 0; i < greater_than_constraints.size(); i++, current_constraint_index++) {
    ret[current_constraint_index] =
        expressionFromCoeffs(greater_than_constraints[i].head(num_decision_vars).eval(), "x");
    ret[current_constraint_index] += std::format(" ≥ {}", greater_than_constraints[i](num_decision_vars));
  }
  return ret;
}
