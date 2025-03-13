// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/NonlinearProblem.h"

#include <utility>

namespace suboptimal {
Variable& NonlinearProblem::makeDecisionVariable() {
  decision_vars.emplace_back();
  return decision_vars.back();
}

void NonlinearProblem::minimize(const Variable& objective) {
  this->objective = objective;
}

void NonlinearProblem::minimize(Variable&& objective) {
  this->objective = std::move(objective);
}

void NonlinearProblem::maximize(const Variable& objective) {
  this->objective = -objective;
}

void NonlinearProblem::maximize(Variable&& objective) {
  this->objective = -std::move(objective);
}

void NonlinearProblem::addConstraint(const Constraint& constraint) {
  if (constraint.is_equality) {
    equality_constraints.push_back(constraint.var);
  } else {
    inequality_constraints.push_back(constraint.var);
  }
}

void NonlinearProblem::addConstraint(Constraint&& constraint) {
  if (constraint.is_equality) {
    equality_constraints.push_back(std::move(constraint.var));
  } else {
    inequality_constraints.push_back(std::move(constraint.var));
  }
}
}  // namespace suboptimal
