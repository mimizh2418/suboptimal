// Copyright (c) 2024 Alvin Zhang.

#pragma once

#include <chrono>
#include <cmath>

#include <Eigen/Core>

struct MotorConstants {
  MotorConstants(const double nominal_voltage, const double stall_torque_Nm, const double stall_current_A,
                 const double free_speed_radPerS, const double free_current_A)
      : nominal_voltage(nominal_voltage),
        stall_torque_Nm(stall_torque_Nm),
        stall_current_A(stall_current_A),
        free_speed_radPerS(free_speed_radPerS),
        free_current_A(free_current_A),
        r_ohms(nominal_voltage / stall_current_A),
        kV_radPerSPerV(free_speed_radPerS / (nominal_voltage - r_ohms * free_current_A)),
        kT_NmPerA(stall_torque_Nm / stall_current_A) {}

  const double nominal_voltage;
  const double stall_torque_Nm;
  const double stall_current_A;
  const double free_speed_radPerS;
  const double free_current_A;
  const double r_ohms;
  const double kV_radPerSPerV;
  const double kT_NmPerA;
};

struct ElevatorDynamics {
  ElevatorDynamics(const double carriage_mass_kg, const double drum_radius_m, const double gearing,
                   const double min_position_m, const double max_position_m, const std::chrono::duration<double> dt,
                   const MotorConstants& motor)
      : A{{0, 1},  //
          {0, (-std::pow(gearing, 2) * motor.kT_NmPerA) /
                  (motor.r_ohms * std::pow(drum_radius_m, 2) * carriage_mass_kg * motor.kV_radPerSPerV)}},
        B{{0, (gearing * motor.kT_NmPerA) / (motor.r_ohms * drum_radius_m * carriage_mass_kg)}},
        A_d{(Eigen::Matrix2d::Identity() + A * dt.count()).eval()},
        B_d{(B * dt.count()).eval()},
        max_input_V(motor.nominal_voltage),
        min_position_m(min_position_m),
        max_position_m(max_position_m) {}

  Eigen::Vector2d simulateTimestep(const Eigen::Vector2d& state, const double voltage) const {
    Eigen::Vector2d new_state = A_d * state + B_d * voltage;
    new_state(0) = std::clamp(new_state(0), min_position_m, max_position_m);
    return new_state;
  }

  const Eigen::Matrix2d A;
  const Eigen::Vector2d B;
  const Eigen::Matrix2d A_d;
  const Eigen::Vector2d B_d;
  const double max_input_V;
  const double min_position_m;
  const double max_position_m;
};
