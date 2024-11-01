// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <gsl/assert>

constexpr double EPSILON = 1e-10;
constexpr double ABS_THRESH = 1e-10;

template <typename T>
concept ComparableFloat = std::is_floating_point_v<T>;

template <ComparableFloat T>
bool isApprox(T a, T b, T epsilon = EPSILON, T abs_thresh = ABS_THRESH) {
  Expects(epsilon >= std::numeric_limits<T>::epsilon() && epsilon <= 1);
  Expects(abs_thresh >= 0);

  if (a == b) {
    return true;
  }
  T norm = std::min(std::abs(a + b), std::numeric_limits<T>::max());
  return std::abs(a - b) <= std::max(abs_thresh, epsilon * norm);
}

template <ComparableFloat T>
bool approxLEQ(T a, T b, T epsilon = EPSILON, T abs_thresh = ABS_THRESH) {
  return a < b || isApprox<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxGEQ(T a, T b, T epsilon = EPSILON, T abs_thresh = ABS_THRESH) {
  return a > b || isApprox<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxLT(T a, T b, T epsilon = EPSILON, T abs_thresh = ABS_THRESH) {
  return !approxGEQ<T>(a, b, epsilon, abs_thresh);
}

template <ComparableFloat T>
bool approxGT(T a, T b, T epsilon = EPSILON, T abs_thresh = ABS_THRESH) {
  return !approxLEQ<T>(a, b, epsilon, abs_thresh);
}
