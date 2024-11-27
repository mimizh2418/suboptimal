#include "suboptimal/autodiff/derivatives.h"

#include <Eigen/Core>

namespace suboptimal {
VectorXv gradient(const Variable& var, const Eigen::Ref<const VectorXv>& wrt) {
  var.expr->updateChildren();

  for (int i = 0; i < wrt.size(); i++) {
    wrt(i).expr->wrt_index = i;
  }

  // Initialize adjoint expressions
  for (const auto expr : var.expr->children) {
    expr->adjoint_expr = std::make_shared<Expression>(0.0);
  }
  // Root adjoint is always 1
  var.expr->adjoint_expr = std::make_shared<Expression>(1.0);

  // Compute adjoint expressions
  VectorXv grad{wrt.size()};
  for (const auto expr : var.expr->children) {
    if (expr->lhs != nullptr && !expr->lhs->constEquals(0.0)) {
      expr->lhs->adjoint_expr =
          expr->lhs->adjoint_expr + expr->lhs_adjoint_expr_func(expr->lhs, expr->rhs, expr->adjoint_expr);
    }
    if (expr->rhs != nullptr && !expr->rhs->constEquals(0.0)) {
      expr->rhs->adjoint_expr =
          expr->rhs->adjoint_expr + expr->rhs_adjoint_expr_func(expr->lhs, expr->rhs, expr->adjoint_expr);
    }
    if (expr->wrt_index != -1) {
      grad(expr->wrt_index) = Variable{expr->adjoint_expr};
    }
  }

  // Reset adjoint expressions and wrt indices
  for (const auto expr : var.expr->children) {
    if (expr->lhs != nullptr) {
      expr->lhs->adjoint_expr = nullptr;
    }
    if (expr->rhs != nullptr) {
      expr->rhs->adjoint_expr = nullptr;
    }
  }
  for (int i = 0; i < wrt.size(); i++) {
    wrt(i).expr->wrt_index = -1;
  }

  return grad;
}

Variable derivative(const Variable& var, const Variable& wrt) {
  return gradient(var, VectorXv{{wrt}})(0);
}
}  // namespace suboptimal
