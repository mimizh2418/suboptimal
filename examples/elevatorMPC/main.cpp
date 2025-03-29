// Copyright (c) 2024 Alvin Zhang.

#include <suboptimal/solvers/ExitStatus.h>

#include <chrono>
#include <iostream>

#include <Eigen/Core>

#include "Dynamics.h"
#include "ElevatorMPC.h"

int main() {
  using namespace std::chrono_literals;

  constexpr std::chrono::duration<double> dt = 10ms;

  // Motor characteristics (Falcon 500 w/FOC)
  constexpr double nominal_voltage = 12.0;
  constexpr double stall_torque_Nm = 5.84;
  constexpr double stall_current_A = 304.0;
  constexpr double free_current_A = 1.5;
  constexpr double free_speed_radPerS = 636.6961104;
  constexpr MotorConstants motor_constants{nominal_voltage, stall_torque_Nm, stall_current_A, free_speed_radPerS,
                                           free_current_A};

  // Elevator characteristics
  constexpr double carriage_mass_kg = 15.0;
  constexpr double drum_radius_m = 0.01905;
  constexpr double gearing = 4.64;
  constexpr double min_position_m = 0.0;
  constexpr double max_position_m = 1.0;
  const ElevatorDynamics elevator_dynamics{carriage_mass_kg, drum_radius_m, gearing, min_position_m, max_position_m, dt,
                                           motor_constants};

  // MPC setup
  const Eigen::Vector2d weights{1.0, 0.1};
  constexpr int horizon = 15;
  const ElevatorMPC controller{elevator_dynamics, weights, horizon};

  const Eigen::Vector2d reference{0.5, 0.0};
  Eigen::Vector2d state{0.0, 0.0};

  while (!((state - reference).cwiseAbs().array() < 1e-2).all()) {
    auto [voltage, status] = controller.calculate(state, reference);
    if (status != suboptimal::ExitStatus::Success) {
      std::cout << "Solver failed with status: " << suboptimal::toString(status) << std::endl;
      break;
    }

    state = elevator_dynamics.simulateTimestep(state, voltage);
    std::cout << "Position: " << state(0) << "; Velocity: " << state(1) << "; Voltage: " << voltage << std::endl;
  }

  return 0;
}
