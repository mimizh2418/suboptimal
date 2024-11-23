// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <memory>

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
  ExpressionPtr adjointExpr = nullptr;

  ExpressionPtr lhs = nullptr;
  ExpressionPtr rhs = nullptr;

  ValueFunc valueFunc = nullptr;  // Function giving the value of the expression

  AdjointValueFunc lhsAdjointValueFunc = nullptr;  // Function giving the adjoint value of the LHS expression
  AdjointValueFunc rhsAdjointValueFunc = nullptr;  // Function giving the adjoint value of the RHS expression

  AdjointExprFunc lhsAdjointExprFunc = nullptr;  // Function giving the adjoint expression of the LHS expression
  AdjointExprFunc rhsAdjointExprFunc = nullptr;  // Function giving the adjoint expression of the RHS expression

  ExpressionType type = ExpressionType::Constant;

  /**
   * Constructs a nullary expression
   */
  explicit Expression(double value, ExpressionType type = ExpressionType::Constant);

  /**
   * Constructs a unary expression
   */
  Expression(ExpressionType type, ValueFunc valueFunc, AdjointValueFunc adjointValueFunc,
             AdjointExprFunc adjointExprFunc, ExpressionPtr arg);

  /**
   * Constructs a binary expression
   */
  Expression(ExpressionType type, ValueFunc valueFunc, AdjointValueFunc lhsAdjointValueFunc,
             AdjointValueFunc rhsAdjointValueFunc, AdjointExprFunc lhsAdjointExprFunc,
             AdjointExprFunc rhsAdjointExprFunc, ExpressionPtr lhs, ExpressionPtr rhs);

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
   * Updates the value of the expression, traversing the expression tree and updating all expressions this expression
   * depends on
   */
  void update();
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
