// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/autodiff/Expression.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <ranges>
#include <vector>

namespace suboptimal {
Expression::Expression(const double value, const ExpressionType type) : value{value}, type{type} {}

Expression::Expression(const ExpressionType type, const ValueFunc valueFunc, const AdjointValueFunc adjointValueFunc,
                       const AdjointExprFunc adjointExprFunc, const ExpressionPtr arg)  // NOLINT
    : value{valueFunc(arg->value, 0.0)},
      lhs{arg},
      valueFunc{valueFunc},
      lhsAdjointValueFunc{adjointValueFunc},
      lhsAdjointExprFunc{adjointExprFunc},
      type{type} {}

Expression::Expression(const ExpressionType type, const ValueFunc valueFunc, const AdjointValueFunc lhsAdjointValueFunc,
                       const AdjointValueFunc rhsAdjointValueFunc, const AdjointExprFunc lhsAdjointExprFunc,
                       const AdjointExprFunc rhsAdjointExprFunc, const ExpressionPtr lhs,  // NOLINT
                       const ExpressionPtr rhs)                                            // NOLINT
    : value{valueFunc(lhs->value, rhs->value)},
      lhs{lhs},
      rhs{rhs},
      valueFunc{valueFunc},
      lhsAdjointValueFunc{lhsAdjointValueFunc},
      rhsAdjointValueFunc{rhsAdjointValueFunc},
      lhsAdjointExprFunc{lhsAdjointExprFunc},
      rhsAdjointExprFunc{rhsAdjointExprFunc},
      type{type} {}

void Expression::update() {
  if (isIndependent()) {
    // Expression represents either a constant or an independent variable, so no update is needed
    return;
  }

  std::vector<Expression*> exprs{};
  std::vector<Expression*> stack{};

  if (lhs != nullptr) {
    stack.push_back(lhs.get());
  }
  if (rhs != nullptr) {
    stack.push_back(rhs.get());
  }
  while (!stack.empty()) {
    const auto expr = stack.back();
    stack.pop_back();
    exprs.push_back(expr);

    // Don't need to update the values of independent variables
    if (expr->isIndependent()) {
      continue;
    }

    if (expr->lhs != nullptr) {
      stack.push_back(expr->lhs.get());
    }
    if (expr->rhs != nullptr) {
      stack.push_back(expr->rhs.get());
    }
  }

  for (const auto expr : std::ranges::reverse_view(exprs)) {
    if (expr->isUnary()) {
      expr->value = expr->valueFunc(expr->lhs->value, 0.0);
    } else if (expr->isBinary()) {
      expr->value = expr->valueFunc(expr->lhs->value, expr->rhs->value);
    }
  }
  value = valueFunc(lhs->value, rhs->value);
}

// operator overloading boilerplate :skull:
// TODO operator null checks

ExpressionPtr operator+(const ExpressionPtr& x) {
  return x;
}

ExpressionPtr operator-(const ExpressionPtr& x) {
  if (x->isConstant()) {
    return std::make_shared<Expression>(-(x->value));
  }

  return std::make_shared<Expression>(
      x->type, [](const double val, double) { return -val; },
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
      std::max(lhs->type, rhs->type), [](const double lhs_val, const double rhs_val) { return lhs_val + rhs_val; },
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
      std::max(lhs->type, rhs->type), [](const double lhs_val, const double rhs_val) { return lhs_val - rhs_val; },
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

  ExpressionType type;
  if (lhs->isConstant()) {
    type = rhs->type;
  } else if (rhs->isConstant()) {
    type = lhs->type;
  } else if (lhs->type == ExpressionType::Linear && rhs->type == ExpressionType::Linear) {
    type = ExpressionType::Quadratic;
  } else {
    type = ExpressionType::Nonlinear;
  }

  return std::make_shared<Expression>(
      type, [](const double lhs_val, const double rhs_val) { return lhs_val * rhs_val; },
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

ExpressionPtr func(const ExpressionPtr& x) {
  return x;
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
      rhs->isConstant() ? lhs->type : ExpressionType::Nonlinear,
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::abs(val); },
      [](const double val, double, const double parent_adjoint) { return val < 0 ? -parent_adjoint : parent_adjoint; },
      [](const ExpressionPtr& expr, const ExpressionPtr&, const ExpressionPtr& parent_adjoint) {
        return expr->value < 0 ? -parent_adjoint : parent_adjoint;
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::sqrt(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::exp(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::log(val); },
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

  ExpressionType type;
  if (base->type == ExpressionType::Linear && exponent->constEquals(2.0)) {
    type = ExpressionType::Quadratic;
  } else {
    type = ExpressionType::Nonlinear;
  }

  return std::make_shared<Expression>(
      type, [](const double base_val, const double exp_val) { return std::pow(base_val, exp_val); },
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
      ExpressionType::Nonlinear, [](const double x_val, const double y_val) { return std::hypot(x_val, y_val); },
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

ExpressionPtr sin(const ExpressionPtr& x) {
  if (x->constEquals(0.0)) {
    return x;
  }
  if (x->isConstant()) {
    return std::make_shared<Expression>(std::sin(x->value));
  }

  return std::make_shared<Expression>(
      ExpressionType::Nonlinear, [](const double val, double) { return std::sin(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::cos(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::tan(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::asin(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::acos(val); },
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
      ExpressionType::Nonlinear, [](const double val, double) { return std::atan(val); },
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
      ExpressionType::Nonlinear, [](const double y_val, const double x_val) { return std::atan2(y_val, x_val); },
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
}  // namespace suboptimal
