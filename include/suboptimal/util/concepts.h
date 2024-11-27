// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <type_traits>

namespace suboptimal {
struct Variable;

template <typename T>
concept VariableLike = std::is_same_v<T, Variable> || std::is_arithmetic_v<T>;
}  // namespace suboptimal
