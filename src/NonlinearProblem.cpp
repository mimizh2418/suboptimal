#include "suboptimal/NonlinearProblem.h"

namespace suboptimal {
NonlinearProblem::NonlinearProblem() = default;

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
  constraints.push_back(constraint);
}

void NonlinearProblem::addConstraint(Constraint&& constraint) {
  constraints.push_back(std::move(constraint));
}
}
