#pragma once

#include <Eigen/Core>

#include "suboptimal/autodiff/Variable.h"

namespace suboptimal {
VectorXv gradient(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);

Variable derivative(const Variable& var, const Variable& wrt);

MatrixXv jacobian(const Eigen::Ref<const VectorXv>& vars, const Eigen::Ref<const VectorXv>& wrt);

MatrixXv hessian(const Variable& var, const Eigen::Ref<const VectorXv>& wrt);
}  // namespace suboptimal