// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <chrono>
#include <gsl/assert>

class SolverProfiler {
 public:
  void startIteration() {
    current_iteration_start_time = std::chrono::system_clock::now();
    if (!started) {
      started = true;
      solve_start_time = current_iteration_start_time;
    }
    iteration_started = true;
  }

  void endIteration() {
    if (!started || !iteration_started) {
      return;
    }

    last_iteration_end_time = std::chrono::system_clock::now();

    const auto iteration_time = last_iteration_end_time - current_iteration_start_time;
    avg_iteration_time = (avg_iteration_time * solver_iterations + iteration_time) / (solver_iterations + 1);

    solver_iterations++;
    iteration_started = false;
  }

  std::chrono::duration<double, std::milli> totalSolveTime() const {
    if (!started) {
      return std::chrono::duration<double, std::milli>(0.0);
    }
    return last_iteration_end_time - solve_start_time;
  }

  std::chrono::duration<double, std::milli> avgIterationTime() const { 
    if (!started) {
      return std::chrono::duration<double, std::milli>(0.0);
    }
    return avg_iteration_time; 
  }

  int numIterations() const { return solver_iterations; }

 private:
  bool started = false;
  std::chrono::system_clock::time_point solve_start_time;

  int solver_iterations = 0;
  bool iteration_started = false;

  std::chrono::system_clock::time_point current_iteration_start_time;
  std::chrono::system_clock::time_point last_iteration_end_time;

  std::chrono::duration<double, std::milli> avg_iteration_time{0.0};
};
