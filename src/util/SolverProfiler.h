#pragma once

#include <chrono>
#include <Eigen/Sparse>

class SolverProfiler {
 public:
  void startIteration() { current_iteration_start_time = std::chrono::system_clock::now(); }

  void endIteration() {
    const auto current_time = std::chrono::system_clock::now();
    const auto iteration_time = current_time - current_iteration_start_time;
    avg_iteration_time = (avg_iteration_time * solver_iterations + iteration_time) / (solver_iterations + 1);
    solver_iterations++;
  }

  std::chrono::duration<double, std::milli> getAvgIterationTime() const { return avg_iteration_time; }

  int getNumIterations() const { return solver_iterations; }

 private:
  int solver_iterations = 0;
  std::chrono::duration<double, std::milli> avg_iteration_time{0.0};
  std::chrono::system_clock::time_point current_iteration_start_time;
};
