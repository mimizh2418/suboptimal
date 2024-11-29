// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <memory>
#include <vector>

#include "suboptimal/autodiff/ExpressionType.h"

namespace suboptimal {
struct Expression;
using ExpressionPtr = std::shared_ptr<Expression>;

/**
 * Autodiff expression node that represents a nullary, unary, or binary operation
 */
struct Expression {
  // LHS, RHS
  using ValueFunc = double (*)(double, double);
  // LHS, RHS, parent adjoint: adj(parent) * ∂(parent) / ∂(self)
  using AdjointValueFunc = double (*)(double, double, double);
  // LHS, RHS, parent adjoint: adj(parent) * ∂(parent) / ∂(self)
  using AdjointExprFunc = ExpressionPtr (*)(const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr&);

  double value = 0.0;
  double adjoint = 0.0;
  ExpressionPtr adjoint_expr = nullptr;

  ExpressionPtr lhs = nullptr;
  ExpressionPtr rhs = nullptr;

  ValueFunc value_func = nullptr;  // Function giving the value of the expression

  AdjointValueFunc lhs_adjoint_value = nullptr;       // Function giving the adjoint value of the LHS expression
  AdjointValueFunc rhs_adjoint_value_func = nullptr;  // Function giving the adjoint value of the RHS expression

  AdjointExprFunc lhs_adjoint_expr_func = nullptr;  // Function giving the adjoint expression of the LHS expression
  AdjointExprFunc rhs_adjoint_expr_func = nullptr;  // Function giving the adjoint expression of the RHS expression

  ExpressionType type = ExpressionType::Constant;

  int wrt_index = -1;

  int indegree = 0;
  std::vector<Expression*> children{};  // Children of this expression sorted from parent to child, excluding constants.
                                        // Can be raw pointers since shared_ptrs to the children are owned by the parent

  /**
   * Constructs a nullary expression
   */
  explicit Expression(double value, ExpressionType type = ExpressionType::Constant);

  /**
   * Constructs a unary expression
   */
  Expression(ExpressionType type, ValueFunc value_func, AdjointValueFunc adjoint_value_func,
             AdjointExprFunc adjoint_expr_func, ExpressionPtr arg);

  /**
   * Constructs a binary expression
   */
  Expression(ExpressionType type, ValueFunc valueFunc, AdjointValueFunc lhs_adjoint_value_func,
             AdjointValueFunc rhs_adjoint_value_func, AdjointExprFunc lhs_adjoint_expr_func,
             AdjointExprFunc rhs_adjoint_expr_func, ExpressionPtr lhs, ExpressionPtr rhs);

  /**
   * Checks if the value of this expression is independent of any other expressions
   */
  bool isIndependent() const { return lhs == nullptr && rhs == nullptr; }

  /**
   * Checks if the value of this expression is constant
   */
  bool isConstant() const { return isIndependent() && type == ExpressionType::Constant; }

  /**
   * Checks if the expression represents a unary operation
   */
  bool isUnary() const { return !isIndependent() && rhs == nullptr; }

  /**
   * Checks if the expression represents a binary operation
   */
  bool isBinary() const { return !isIndependent() && rhs != nullptr; }

  /**
   * Checks if the expression is a constant with a specific value
   */
  bool constEquals(const double val) const { return isConstant() && value == val; }

  /**
   * Updates the list of child nodes of this expression
   */
  void updateChildren();

  /**
   * Updates the value of the expression, traversing the expression tree and updating all expressions this expression
   * depends on
   */
  void updateValue();
};

// Arithmetic operator overloads

ExpressionPtr operator+(const ExpressionPtr& x);
ExpressionPtr operator-(const ExpressionPtr& x);

ExpressionPtr operator+(const ExpressionPtr& lhs, const ExpressionPtr& rhs);
ExpressionPtr operator-(const ExpressionPtr& lhs, const ExpressionPtr& rhs);
ExpressionPtr operator*(const ExpressionPtr& lhs, const ExpressionPtr& rhs);
ExpressionPtr operator/(const ExpressionPtr& lhs, const ExpressionPtr& rhs);

// STL math function overloads

ExpressionPtr abs(const ExpressionPtr& x);
ExpressionPtr sqrt(const ExpressionPtr& x);
ExpressionPtr exp(const ExpressionPtr& x);
ExpressionPtr log(const ExpressionPtr& x);

ExpressionPtr pow(const ExpressionPtr& base, const ExpressionPtr& exponent);
ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y);

ExpressionPtr sin(const ExpressionPtr& x);
ExpressionPtr cos(const ExpressionPtr& x);
ExpressionPtr tan(const ExpressionPtr& x);
ExpressionPtr asin(const ExpressionPtr& x);
ExpressionPtr acos(const ExpressionPtr& x);
ExpressionPtr atan(const ExpressionPtr& x);
ExpressionPtr atan2(const ExpressionPtr& y, const ExpressionPtr& x);
}  // namespace suboptimal
