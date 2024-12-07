// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/NonlinearProblem.h"

#include <utility>

namespace suboptimal {
Variable& NonlinearProblem::decisionVariable() {
  decision_vars.emplace_back();
  return decision_vars.back();
}

void NonlinearProblem::minimize(const Variable& objective) {
  this->objective = objective;
}

void NonlinearProblem::minimize(Variable&& objective) {
  this->objective = std::move(objective);
}

void NonlinearProblem::addConstraint(const Constraint& constraint) {
  if (constraint.is_equality) {
    equality_constraints.push_back(constraint);
  } else {
    inequality_constraints.push_back(constraint);
  }
}

void NonlinearProblem::addConstraint(Constraint&& constraint) {
  if (constraint.is_equality) {
    equality_constraints.push_back(std::move(constraint));
  } else {
    inequality_constraints.push_back(std::move(constraint));
  }
}
}  // namespace suboptimal
