// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/autodiff/Variable.h"

#include <memory>
#include <utility>

namespace suboptimal {
Variable::Variable(double value) : expr{std::make_shared<Expression>(value, ExpressionType::Linear)} {}

Variable::Variable(const ExpressionPtr& expr) : expr{expr} {}

Variable::Variable(ExpressionPtr&& expr) : expr{std::move(expr)} {}

void Variable::update() {
  expr->update();
}

bool Variable::setValue(const double value) {
  if (expr->isIndependent()) {
    expr->value = value;
    return true;
  }
  return false;
}

// Behold! Accursed operator overloads!

Variable& Variable::operator+=(const Variable& other) {
  *this = *this + other;
  return *this;
}

Variable& Variable::operator-=(const Variable& other) {
  *this = *this - other;
  return *this;
}

Variable& Variable::operator*=(const Variable& other) {
  *this = *this * other;
  return *this;
}

Variable& Variable::operator/=(const Variable& other) {
  *this = *this / other;
  return *this;
}

Variable& Variable::operator+=(const double other) {
  *this = *this + other;
  return *this;
}

Variable& Variable::operator-=(const double other) {
  *this = *this - other;
  return *this;
}

Variable& Variable::operator*=(const double other) {
  *this = *this * other;
  return *this;
}

Variable& Variable::operator/=(const double other) {
  *this = *this / other;
  return *this;
}

Variable operator+(const Variable& x) {
  return x;
}

Variable operator-(const Variable& x) {
  return {-x.expr};
}

Variable operator+(const Variable& lhs, const Variable& rhs) {
  return {lhs.expr + rhs.expr};
}

Variable operator-(const Variable& lhs, const Variable& rhs) {
  return {lhs.expr - rhs.expr};
}

Variable operator*(const Variable& lhs, const Variable& rhs) {
  return {lhs.expr * rhs.expr};
}

Variable operator/(const Variable& lhs, const Variable& rhs) {
  return {lhs.expr / rhs.expr};
}

Variable operator+(const double lhs, const Variable& rhs) {
  return {std::make_shared<Expression>(lhs) + rhs.expr};
}

Variable operator-(const double lhs, const Variable& rhs) {
  return {std::make_shared<Expression>(lhs) - rhs.expr};
}

Variable operator*(const double lhs, const Variable& rhs) {
  return {std::make_shared<Expression>(lhs) * rhs.expr};
}

Variable operator/(const double lhs, const Variable& rhs) {
  return {std::make_shared<Expression>(lhs) / rhs.expr};
}

Variable operator+(const Variable& lhs, const double rhs) {
  return {lhs.expr + std::make_shared<Expression>(rhs)};
}

Variable operator-(const Variable& lhs, const double rhs) {
  return {lhs.expr - std::make_shared<Expression>(rhs)};
}

Variable operator*(const Variable& lhs, const double rhs) {
  return {lhs.expr - std::make_shared<Expression>(rhs)};
}

Variable operator/(const Variable& lhs, const double rhs) {
  return {lhs.expr - std::make_shared<Expression>(rhs)};
}

Variable abs(const Variable& x) {
  return {abs(x.expr)};
}

Variable sqrt(const Variable& x) {
  return {sqrt(x.expr)};
}

Variable exp(const Variable& x) {
  return {exp(x.expr)};
}

Variable log(const Variable& x) {
  return {log(x.expr)};
}

Variable pow(const Variable& base, const Variable& exponent) {
  return {pow(base.expr, exponent.expr)};
}

Variable hypot(const Variable& x, const Variable& y) {
  return {hypot(x.expr, y.expr)};
}

Variable sin(const Variable& x) {
  return {sin(x.expr)};
}
Variable cos(const Variable& x) {
  return {cos(x.expr)};
}

Variable tan(const Variable& x) {
  return {tan(x.expr)};
}

Variable asin(const Variable& x) {
  return {asin(x.expr)};
}

Variable acos(const Variable& x) {
  return {acos(x.expr)};
}

Variable atan(const Variable& x) {
  return {atan(x.expr)};
}

Variable atan2(const Variable& y, const Variable& x) {
  return {atan2(y.expr, x.expr)};
}
}  // namespace suboptimal
