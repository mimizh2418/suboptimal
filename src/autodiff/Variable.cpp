// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/autodiff/Variable.h"

#include <memory>
#include <utility>

namespace suboptimal {
Variable::Variable(double value) : expr{std::make_shared<Expression>(value, Linearity::Linear)} {}

Variable::Variable(const ExpressionPtr& expr) : expr{expr} {}

Variable::Variable(ExpressionPtr&& expr) : expr{std::move(expr)} {}

Variable Variable::Constant(double value) {
  return std::make_shared<Expression>(value);
}

void Variable::updateGraph() const {
  expr->updateChildren();
}

void Variable::updateValue() const {
  expr->updateValue();
}

bool Variable::setValue(const double value) {
  if (expr->isIndependent()) {
    expr->value = value;
    return true;
  }
  return false;
}

double Variable::getValue() const {
  updateValue();
  return expr->value;
}

Variable operator+(const Variable& x) {
  return x;
}

Variable operator-(const Variable& x) {
  return -x.expr;
}

Variable abs(const Variable& x) {
  return abs(x.expr);
}

Variable sqrt(const Variable& x) {
  return sqrt(x.expr);
}

Variable exp(const Variable& x) {
  return exp(x.expr);
}

Variable log(const Variable& x) {
  return log(x.expr);
}

Variable erf(const Variable& x) {
  return erf(x.expr);
}

Variable sin(const Variable& x) {
  return sin(x.expr);
}
Variable cos(const Variable& x) {
  return cos(x.expr);
}

Variable tan(const Variable& x) {
  return tan(x.expr);
}

Variable asin(const Variable& x) {
  return asin(x.expr);
}

Variable acos(const Variable& x) {
  return acos(x.expr);
}

Variable atan(const Variable& x) {
  return atan(x.expr);
}

Variable sinh(const Variable& x) {
  return sinh(x.expr);
}

Variable cosh(const Variable& x) {
  return cosh(x.expr);
}

Variable tanh(const Variable& x) {
  return tanh(x.expr);
}

Variable asinh(const Variable& x) {
  return asinh(x.expr);
}

Variable acosh(const Variable& x) {
  return acosh(x.expr);
}

Variable atanh(const Variable& x) {
  return atanh(x.expr);
}
}  // namespace suboptimal
