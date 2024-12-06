// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <concepts>
#include <utility>
#include <vector>

#include "suboptimal/autodiff/Variable.h"

namespace suboptimal {

/**
 * Represents a nonlinear constraint in standard form (c_i(x) >= 0 or c_e(x) == 0)
 */
struct Constraint {
  Variable var;
  bool is_equality;

  template <VariableLike LHS, VariableLike RHS>
    requires std::same_as<LHS, Variable> || std::same_as<RHS, Variable>
  Constraint(Variable&& lhs, Variable&& rhs, const int type)
      : var{(type == -1 ? -1 : 1) * (lhs - rhs)}, is_equality{type == 0} {}
};

template <VariableLike LHS, VariableLike RHS>
  requires std::same_as<LHS, Variable> || std::same_as<RHS, Variable>
Constraint operator==(LHS&& lhs, RHS&& rhs) {
  return Constraint{std::forward<LHS>(lhs), std::forward<RHS>(rhs), 0};
}

template <VariableLike LHS, VariableLike RHS>
  requires std::same_as<LHS, Variable> || std::same_as<RHS, Variable>
Constraint operator<=(LHS&& lhs, RHS&& rhs) {
  return Constraint{std::forward<LHS>(lhs), std::forward<RHS>(rhs), -1};
}

template <VariableLike LHS, VariableLike RHS>
  requires std::same_as<LHS, Variable> || std::same_as<RHS, Variable>
Constraint operator>=(LHS&& lhs, RHS&& rhs) {
  return Constraint{std::forward<LHS>(lhs), std::forward<RHS>(rhs), 1};
}

/**
 * Represents a nonlinear optimization problem in standard form:
 * min f(x),
 * s.t. c_i(x) >= 0, c_e(x) == 0
 */
class NonlinearProblem {
 public:
  /**
   * Creates a nonlinear optimization problem
   */
  NonlinearProblem();

  /**
   * Adds a decision variable to the problem
   */
  Variable& decisionVariable();

  /**
   * Sets the objective function to be minimized
   * @param objective the function to minimize
   */
  void minimize(const Variable& objective);

  /**
   * Sets the objective function to be minimized
   * @param objective the function to minimize
   */
  void minimize(Variable&& objective);

  /**
   * Sets the objective function to be maximized
   * @param objective the function to maximize
   */
  void maximize(const Variable& objective);

  /**
   * Sets the objective function to be maximized
   * @param objective the function to maximize
   */
  void maximize(Variable&& objective);

  /**
   * Adds a constraint to the optimization problem
   * @param constraint the constraint to add
   */
  void addConstraint(const Constraint& constraint);

  /**
   * Adds a constraint to the optimization problem
   * @param constraint the constraint to add
   */
  void addConstraint(Constraint&& constraint);

 private:
  Variable objective;
  std::vector<Variable> decision_vars;
  std::vector<Constraint> inequality_constraints;
  std::vector<Constraint> equality_constraints;
};
}  // namespace suboptimal
