// Copyright (c) 2024 Alvin Zhang.

#include "suboptimal/autodiff/derivatives.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <Eigen/Core>

namespace suboptimal {
Gradient::Gradient(const Variable& var, const Eigen::Ref<const VectorXv>& wrt) : var{var}, wrt{wrt}, value(wrt.size()) {
  std::ranges::for_each(wrt, [](const Variable& v) { v.expr->adjoint = 0.0; });
  var.expr->updateAdjoints();
  for (int i = 0; i < wrt.size(); i++) {
    value.coeffRef(i) = wrt(i).expr->adjoint;
  }
}

const Eigen::SparseVector<double>& Gradient::getValue() {
  if (var.getLinearity() > Linearity::Linear) {
    std::ranges::for_each(wrt, [](const Variable& v) { v.expr->adjoint = 0.0; });
    var.expr->updateAdjoints();
    for (int i = 0; i < wrt.size(); i++) {
      value.coeffRef(i) = wrt(i).expr->adjoint;
    }
  }
  return value;
}

VectorXv Gradient::getExpr() {
  if (var.getLinearity() <= Linearity::Linear) {
    VectorXv grad{wrt.size()};
    for (int i = 0; i < wrt.size(); i++) {
      grad(i) = Variable{std::make_shared<Expression>(0.0)};
    }
    for (Eigen::SparseVector<double>::InnerIterator it(value); it; ++it) {
      grad(it.index()) = Variable{std::make_shared<Expression>(it.value())};
    }
  }

  var.expr->updateChildren();

  // Initialize adjoint expressions
  std::ranges::for_each(var.expr->children,
                        [](Expression* expr) { expr->adjoint_expr = std::make_shared<Expression>(0.0); });
  // Root adjoint is always 1
  var.expr->adjoint_expr->value = 1.0;

  for (const auto expr : var.expr->children) {
    if (expr->lhs != nullptr && !expr->lhs->isConstant()) {
      expr->lhs->adjoint_expr =
          expr->lhs->adjoint_expr + expr->lhs_adjoint_expr_func(expr->lhs, expr->rhs, expr->adjoint_expr);
    }
    if (expr->rhs != nullptr && !expr->rhs->isConstant()) {
      expr->rhs->adjoint_expr =
          expr->rhs->adjoint_expr + expr->rhs_adjoint_expr_func(expr->lhs, expr->rhs, expr->adjoint_expr);
    }
  }

  VectorXv grad{wrt.size()};
  for (int i = 0; i < wrt.size(); i++) {
    if (wrt(i).expr->adjoint_expr != nullptr) {
      grad(i) = Variable{wrt(i).expr->adjoint_expr};
    } else {
      // var is not dependent on wrt(i)
      grad(i) = Variable{std::make_shared<Expression>(0.0)};
    }
  }

  for (const auto expr : var.expr->children) {
    if (expr->lhs != nullptr) {
      expr->lhs->adjoint_expr = nullptr;
    }
    if (expr->rhs != nullptr) {
      expr->rhs->adjoint_expr = nullptr;
    }
  }

  return grad;
}

Derivative::Derivative(const Variable& var, const Variable& wrt)
    : var{var}, wrt{wrt}, gradient{var, Eigen::Vector<Variable, 1>{{wrt}}} {}

double Derivative::getValue() {
  return gradient.getValue().coeff(0);
}

Variable Derivative::getExpr() {
  return gradient.getExpr()(0);
}

Jacobian::Jacobian(const Eigen::Ref<const VectorXv>& vars, const Eigen::Ref<const VectorXv>& wrt)
    : vars{vars}, wrt{wrt}, value(vars.size(), wrt.size()) {
  gradients.reserve(vars.size());
  std::vector<Eigen::Triplet<double>> triplets{};
  for (int i = 0; i < vars.size(); i++) {
    gradients.emplace_back(vars(i), wrt);
    if (vars(i).getLinearity() > Linearity::Linear) {
      nonlinear_rows.push_back(i);
    }

    for (Eigen::SparseVector<double>::InnerIterator it(gradients[i].value); it; ++it) {
      triplets.emplace_back(i, it.index(), it.value());
      if (vars(i).getLinearity() == Linearity::Linear) {
        cache.emplace_back(i, it.index(), it.value());
      }
    }
  }

  value.setFromTriplets(triplets.begin(), triplets.end());
}

const Eigen::SparseMatrix<double>& Jacobian::getValue() {
  if (nonlinear_rows.empty()) {
    return value;
  }

  std::vector<Eigen::Triplet<double>> triplets = cache;
  for (const int row : nonlinear_rows) {
    for (Eigen::SparseVector<double>::InnerIterator it(gradients[row].getValue()); it; ++it) {
      triplets.emplace_back(row, it.index(), it.value());
    }
  }
  value.setFromTriplets(triplets.begin(), triplets.end());

  return value;
}

MatrixXv Jacobian::getExpr() {
  MatrixXv jacobian(vars.size(), wrt.size());
  for (int i = 0; i < vars.size(); i++) {
    jacobian.row(i) = gradients[i].getExpr().transpose();
  }
  return jacobian;
}

Hessian::Hessian(const Variable& var, const Eigen::Ref<const VectorXv>& wrt)
    : jacobian(Gradient{var, wrt}.getExpr(), wrt) {}

const Eigen::SparseMatrix<double>& Hessian::getValue() {
  return jacobian.getValue();
}

MatrixXv Hessian::getExpr() {
  return jacobian.getExpr();
}
}  // namespace suboptimal
