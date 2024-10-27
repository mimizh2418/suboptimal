// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <format>
#include <string>

#include <Eigen/Core>

inline std::string expressionFromCoeffs(const Eigen::VectorXd& coeffs, const std::string& variable_name) {
  std::string ret;
  bool first = true;
  for (Eigen::Index i = 0; i < coeffs.size(); i++) {
    const double coeff = coeffs(i);
    if (coeff == 0) {
      continue;
    }
    if (!first) {
      ret += coeff > 0 ? " + " : " - ";
    } else {
      if (coeff < 0) {
        ret += "-";
      }
      first = false;
    }
    if (std::abs(coeff) != 1) {
      ret += std::format("{}", std::abs(coeff));
    }
    ret += variable_name + "_" + std::to_string(i + 1);
  }
  return ret;
}
