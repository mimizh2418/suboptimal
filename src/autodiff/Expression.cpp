// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/autodiff/Expression.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <ranges>
#include <vector>

namespace suboptimal {
Expression::Expression(const double value, const Linearity linearity) : value{value}, linearity{linearity} {}

Expression::Expression(const Linearity linearity, const ValueFunc value_func, const AdjointValueFunc adjoint_value_func,
                       const AdjointExprFunc adjoint_expr_func, const ExpressionPtr arg)  // NOLINT
    : value{value_func(arg->value, 0.0)},
      lhs{arg},
      value_func{value_func},
      lhs_adjoint_value_func{adjoint_value_func},
      lhs_adjoint_expr_func{adjoint_expr_func},
      linearity{linearity} {}

Expression::Expression(const Linearity linearity, const ValueFunc value_func,
                       const AdjointValueFunc lhs_adjoint_value_func, const AdjointValueFunc rhs_adjoint_value_func,
                       const AdjointExprFunc lhs_adjoint_expr_func, const AdjointExprFunc rhs_adjoint_expr_func,
                       const ExpressionPtr lhs,  // NOLINT
                       const ExpressionPtr rhs)  // NOLINT
    : value{value_func(lhs->value, rhs->value)},
      lhs{lhs},
      rhs{rhs},
      value_func{value_func},
      lhs_adjoint_value_func{lhs_adjoint_value_func},
      rhs_adjoint_value_func{rhs_adjoint_value_func},
      lhs_adjoint_expr_func{lhs_adjoint_expr_func},
      rhs_adjoint_expr_func{rhs_adjoint_expr_func},
      linearity{linearity} {}

void Expression::updateChildren() {
  children.clear();
  if (isIndependent()) {
    // The expression is independent, so no children need to be updated
    return;
  }

  std::vector<Expression*> stack{};

  // Initialize the indegree of all children
  stack.push_back(this);
  while (!stack.empty()) {
    const auto expr = stack.back();
    stack.pop_back();

    // First time visiting this expression, add children to stack
    if (expr->indegree == 0) {
      if (expr->lhs != nullptr && !expr->lhs->isConstant()) {
        stack.push_back(expr->lhs.get());
      }
      if (expr->rhs != nullptr && !expr->rhs->isConstant()) {
        stack.push_back(expr->rhs.get());
      }
    }
    expr->indegree++;
  }

  stack.push_back(this);
  while (!stack.empty()) {
    const auto expr = stack.back();
    stack.pop_back();
    expr->indegree--;

    if (expr->indegree == 0) {
      children.push_back(expr);
      if (expr->lhs != nullptr && !expr->lhs->isConstant()) {
        stack.push_back(expr->lhs.get());
      }
      if (expr->rhs != nullptr && !expr->rhs->isConstant()) {
        stack.push_back(expr->rhs.get());
      }
    }
  }
}

void Expression::updateValue() {
  if (isIndependent()) {
    // Expression represents either a constant or an independent variable, so no update is needed
    return;
  }

  if (children.empty() && (lhs != nullptr || rhs != nullptr)) {
    updateChildren();
  }

  for (const auto expr : std::ranges::reverse_view(children)) {
    if (expr->isUnary()) {
      expr->value = expr->value_func(expr->lhs->value, 0.0);
    } else if (expr->isBinary()) {
      expr->value = expr->value_func(expr->lhs->value, expr->rhs->value);
    }
  }
}

void Expression::updateAdjoints() {
  if (isConstant()) {
    // Constants always have an adjoint of 0
    adjoint = 0.0;
    return;
  }
  updateValue();

  // Initialize adjoints
  std::ranges::for_each(children, [](Expression* expr) { expr->adjoint = 0.0; });
  adjoint = 1.0;

  for (const auto expr : children) {
    if (expr->lhs != nullptr && !expr->lhs->isConstant()) {
      const double expr_rhs = expr->rhs != nullptr ? expr->rhs->value : 0.0;
      expr->lhs->adjoint += expr->lhs_adjoint_value_func(expr->lhs->value, expr_rhs, expr->adjoint);
    }
    if (expr->rhs != nullptr && !expr->rhs->isConstant()) {
      expr->rhs->adjoint += expr->rhs_adjoint_value_func(expr->lhs->value, expr->rhs->value, expr->adjoint);
    }
  }
}

// operator overloading boilerplate

ExpressionPtr operator+(const ExpressionPtr& x) {
  return x;
}

ExpressionPtr operator-(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(-(x->value));
  }

  return std::make_shared<Expression>(
      x->linearity, [](const double val, double) { return -val; },
      [](double, double, const double parent_adjoint) { return -parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) { return -parent_adjoint; },
      x);
}

ExpressionPtr operator+(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  if (lhs->constEquals(0.0)) {
    return rhs;
  }
  if (rhs->constEquals(0.0)) {
    return lhs;
  }
  if (lhs->isConstant() && rhs->isConstant()) {
    return std::make_shared<Expression>(lhs->value + rhs->value);
  }

  return std::make_shared<Expression>(
      std::max(lhs->linearity, rhs->linearity),
      [](const double lhs_val, const double rhs_val) { return lhs_val + rhs_val; },
      [](double, double, const double parent_adjoint) { return parent_adjoint; },
      [](double, double, const double parent_adjoint) { return parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) { return parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) { return parent_adjoint; },
      lhs, rhs);
}

ExpressionPtr operator-(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  if (lhs->constEquals(0.0)) {
    return -rhs;
  }
  if (rhs->constEquals(0.0)) {
    return lhs;
  }
  if (lhs->isConstant() && rhs->isConstant()) {
    return std::make_shared<Expression>(lhs->value - rhs->value);
  }

  return std::make_shared<Expression>(
      std::max(lhs->linearity, rhs->linearity),
      [](const double lhs_val, const double rhs_val) { return lhs_val - rhs_val; },
      [](double, double, const double parent_adjoint) { return parent_adjoint; },
      [](double, double, const double parent_adjoint) { return -parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) { return parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) { return -parent_adjoint; },
      lhs, rhs);
}

ExpressionPtr operator*(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  if (lhs->constEquals(0.0) || rhs->constEquals(0.0)) {
    return lhs;
  }
  if (lhs->constEquals(1.0)) {
    return rhs;
  }
  if (rhs->constEquals(1.0)) {
    return lhs;
  }
  if (lhs->isConstant() && rhs->isConstant()) {
    return std::make_shared<Expression>(lhs->value * rhs->value);
  }

  Linearity linearity;
  if (lhs->isConstant()) {
    linearity = rhs->linearity;
  } else if (rhs->isConstant()) {
    linearity = lhs->linearity;
  } else if (lhs->linearity == Linearity::Linear && rhs->linearity == Linearity::Linear) {
    linearity = Linearity::Quadratic;
  } else {
    linearity = Linearity::Nonlinear;
  }

  return std::make_shared<Expression>(
      linearity, [](const double lhs_val, const double rhs_val) { return lhs_val * rhs_val; },
      [](double, const double rhs_val, const double parent_adjoint) { return rhs_val * parent_adjoint; },
      [](const double lhs_val, double, const double parent_adjoint) { return lhs_val * parent_adjoint; },
      [](const ExpressionPtr&, const ExpressionPtr& rhs_expr, const ExpressionPtr& parent_adjoint) {
        return rhs_expr * parent_adjoint;
      },
      [](const ExpressionPtr& lhs_expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return lhs_expr * parent_adjoint;
      },
      lhs, rhs);
}

ExpressionPtr operator/(const ExpressionPtr& lhs, const ExpressionPtr& rhs) {
  if (rhs->constEquals(1.0)) {
    return lhs;
  }
  if (lhs->constEquals(0.0)) {
    return lhs;
  }
  if (lhs->isConstant() && rhs->isConstant()) {
    return std::make_shared<Expression>(lhs->value / rhs->value);
  }

  return std::make_shared<Expression>(
      rhs->isConstant() ? lhs->linearity : Linearity::Nonlinear,
      [](const double lhs_val, const double rhs_val) { return lhs_val / rhs_val; },
      [](double, const double rhs_val, const double parent_adjoint) { return parent_adjoint / rhs_val; },
      [](const double lhs_val, const double rhs_val, const double parent_adjoint) {
        return parent_adjoint * -lhs_val / (rhs_val * rhs_val);
      },
      [](const ExpressionPtr&, const ExpressionPtr& rhs_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / rhs_expr;
      },
      [](const ExpressionPtr& lhs_expr, const ExpressionPtr& rhs_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * -lhs_expr / (rhs_expr * rhs_expr);
      },
      lhs, rhs);
}

ExpressionPtr abs(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::abs(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::abs(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * val / std::abs(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * expr / suboptimal::abs(expr);
      },
      x);
}

ExpressionPtr sqrt(const ExpressionPtr& x) {
  if (x->constEquals(0.0) || x->constEquals(1.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::sqrt(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::sqrt(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * 0.5 / std::sqrt(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * std::make_shared<Expression>(0.5) / suboptimal::sqrt(expr);
      },
      x);
}

ExpressionPtr exp(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::exp(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::exp(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * std::exp(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * suboptimal::exp(expr);
      },
      x);
}

ExpressionPtr log(const ExpressionPtr& x) {
  if (x->constEquals(1.0)) {
    std::make_shared<Expression>(0.0);
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::log(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::log(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint / val; },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / expr;
      },
      x);
}

ExpressionPtr pow(const ExpressionPtr& base, const ExpressionPtr& exponent) {
  if (exponent->constEquals(0.0)) {
    return std::make_shared<Expression>(1.0);
  }
  if (exponent->constEquals(1.0)) {
    return base;
  }
  if (base->constEquals(0.0) || base->constEquals(1.0)) {
    return base;
  }
  if (base->isConstant() && exponent->isConstant()) {
    return std::make_shared<Expression>(std::pow(base->value, exponent->value));
  }

  Linearity linearity;
  if (base->linearity == Linearity::Linear && exponent->constEquals(2.0)) {
    linearity = Linearity::Quadratic;
  } else {
    linearity = Linearity::Nonlinear;
  }

  return std::make_shared<Expression>(
      linearity, [](const double base_val, const double exp_val) { return std::pow(base_val, exp_val); },
      [](const double base_val, const double exp_val, const double parent_adjoint) {
        return parent_adjoint * exp_val * std::pow(base_val, exp_val - 1);
      },
      [](const double base_val, const double exp_val, const double parent_adjoint) {
        return parent_adjoint * std::log(base_val) * std::pow(base_val, exp_val);
      },
      [](const ExpressionPtr& base_expr, const ExpressionPtr& exp_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * exp_expr * suboptimal::pow(base_expr, exp_expr - std::make_shared<Expression>(1.0));
      },
      [](const ExpressionPtr& base_expr, const ExpressionPtr& exp_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * suboptimal::log(base_expr) * suboptimal::pow(base_expr, exp_expr);
      },
      base, exponent);
}

ExpressionPtr hypot(const ExpressionPtr& x, const ExpressionPtr& y) {
  if (x->constEquals(0.0)) {
    return abs(y);
  }
  if (y->constEquals(0.0)) {
    return abs(x);
  }
  if (x->isConstant() && y->isConstant()) {
    return std::make_shared<Expression>(std::hypot(x->value, y->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double x_val, const double y_val) { return std::hypot(x_val, y_val); },
      [](const double x_val, const double y_val, const double parent_adjoint) {
        return parent_adjoint * x_val / std::hypot(x_val, y_val);
      },
      [](const double x_val, const double y_val, const double parent_adjoint) {
        return parent_adjoint * y_val / std::hypot(x_val, y_val);
      },
      [](const ExpressionPtr& x_expr, const ExpressionPtr& y_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * x_expr / suboptimal::hypot(x_expr, y_expr);
      },
      [](const ExpressionPtr& x_expr, const ExpressionPtr& y_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * y_expr / suboptimal::hypot(x_expr, y_expr);
      },
      x, y);
}

ExpressionPtr erf(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::erf(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::erf(val); },
      [](const double val, double, const double parent_adjoint) {
        return parent_adjoint * 2.0 * std::exp(-val * val) / std::sqrt(std::numbers::pi);
      },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * std::make_shared<Expression>(2.0) * suboptimal::exp(-expr * expr) /
               std::make_shared<Expression>(std::sqrt(std::numbers::pi));
      },
      x);
}

ExpressionPtr sin(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::sin(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::sin(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * std::cos(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * suboptimal::cos(expr);
      },
      x);
}

ExpressionPtr cos(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::cos(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::cos(val); },
      [](const double val, double, const double parent_adjoint) { return -parent_adjoint * std::sin(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return -parent_adjoint * suboptimal::sin(expr);
      },
      x);
}

ExpressionPtr tan(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::tan(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::tan(val); },
      [](const double val, double, const double parent_adjoint) {
        return parent_adjoint / (std::cos(val) * std::cos(val));
      },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / (suboptimal::cos(expr) * suboptimal::cos(expr));
      },
      x);
}

ExpressionPtr asin(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::asin(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::asin(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint / std::sqrt(1 - val * val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / suboptimal::sqrt(std::make_shared<Expression>(1.0) - expr * expr);
      },
      x);
}

ExpressionPtr acos(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::acos(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::acos(val); },
      [](const double val, double, const double parent_adjoint) { return -parent_adjoint / std::sqrt(1 - val * val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return -parent_adjoint / suboptimal::sqrt(std::make_shared<Expression>(1.0) - expr * expr);
      },
      x);
}

ExpressionPtr atan(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::atan(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::atan(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint / (1 + val * val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / (std::make_shared<Expression>(1.0) + expr * expr);
      },
      x);
}

ExpressionPtr atan2(const ExpressionPtr& y, const ExpressionPtr& x) {
  if (y->constEquals(0.0)) {
    return y;
  }
  if (y->isConstant() && x->isConstant()) {
    return std::make_shared<Expression>(std::atan2(y->value, x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double y_val, const double x_val) { return std::atan2(y_val, x_val); },
      [](const double y_val, const double x_val, const double parent_adjoint) {
        return parent_adjoint * x_val / (y_val * y_val + x_val * x_val);
      },
      [](const double y_val, const double x_val, const double parent_adjoint) {
        return -parent_adjoint * y_val / (y_val * y_val + x_val * x_val);
      },
      [](const ExpressionPtr& y_expr, const ExpressionPtr& x_expr, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * x_expr / (y_expr * y_expr + x_expr * x_expr);
      },
      [](const ExpressionPtr& y_expr, const ExpressionPtr& x_expr, const ExpressionPtr& parent_adjoint) {
        return -parent_adjoint * y_expr / (y_expr * y_expr + x_expr * x_expr);
      },
      y, x);
}

ExpressionPtr sinh(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::sinh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::sinh(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * std::cosh(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * suboptimal::cosh(expr);
      },
      x);
}

ExpressionPtr cosh(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::cosh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::cosh(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint * std::sinh(val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * suboptimal::sinh(expr);
      },
      x);
}

ExpressionPtr tanh(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::tanh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::tanh(val); },
      [](const double val, double, const double parent_adjoint) {
        return parent_adjoint * (1 - std::tanh(val) * std::tanh(val));
      },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint * (std::make_shared<Expression>(1.0) - suboptimal::tanh(expr) * suboptimal::tanh(expr));
      },
      x);
}

ExpressionPtr asinh(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::asinh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::asinh(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint / std::sqrt(1 + val * val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / suboptimal::sqrt(std::make_shared<Expression>(1.0) + expr * expr);
      },
      x);
}

ExpressionPtr acosh(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::acosh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::acosh(val); },
      [](const double val, double, const double parent_adjoint) {
        return parent_adjoint / (std::sqrt(val - 1) * std::sqrt(val + 1));
      },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / (suboptimal::sqrt(expr - std::make_shared<Expression>(1.0)) *
                                 suboptimal::sqrt(expr + std::make_shared<Expression>(1.0)));
      },
      x);
}

ExpressionPtr atanh(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::atanh(x->value));
  }

  return std::make_shared<Expression>(
      Linearity::Nonlinear, [](const double val, double) { return std::atanh(val); },
      [](const double val, double, const double parent_adjoint) { return parent_adjoint / (1 - val * val); },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return parent_adjoint / (std::make_shared<Expression>(1.0) - expr * expr);
      },
      x);
}
}  // namespace suboptimal
