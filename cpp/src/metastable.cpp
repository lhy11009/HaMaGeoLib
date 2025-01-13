#include "metastable.h"
#include <cassert>
#include "IRK4Solver.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>
#include <numeric>



// Function to calculate the first part of the growth rate
double metastable_hosoya_06_eq2_P1(double P, double T, double Coh) {
    const double R = 8.31446;          // J / mol*K, universal gas constant
    const double A = std::exp(-18.0);  // m s-1 wt.ppmH2O^(-3.2)
    const double n = 3.2;
    const double dHa = 274.0e3;        // J / mol, activation enthalpy
    const double Vstar = 3.3e-6;       // m^3 / mol, activation volume

    double growth_rate_part = A * std::pow(Coh, n) * std::exp(-(dHa + P * Vstar) / (R * T));
    return growth_rate_part;
}

// Function to calculate the growth rate considering pressure and temperature variations
double metastable_hosoya_06_eq2(double P, double T, double P_eq, double Coh) {
    const double R = 8.31446;          // J / mol*K, universal gas constant
    const double dV_ol_wd = 2.4e-6;    // m^3 / mol, difference in volume between phases

    assert(P > P_eq && "Pressure must be greater than equilibrium pressure!");

    double dGr = dV_ol_wd * (P - P_eq);
    double growth_rate = metastable_hosoya_06_eq2_P1(P, T, Coh) * T * (1 - std::exp(-dGr / (R * T)));
    return growth_rate;
}

double nucleation_rate_yoshioka_2015(double P, double T, double P_eq) {
    // Constants
    const double gamma = 0.0506;      // J/m^2
    const double K0 = 3.54e38;       // s^-1 m^-2 K^-1
    const double dH_a = 344e3;       // J/mol
    const double V_star = 4e-6;      // m^3/mol
    const double k = 1.38e-23;       // J/K
    const double R = 8.314;          // J/(mol*K)
    const double dV_ol_wd = 2.4e-6;  // m^3/mol
    const double V_initial = 35.17e-6; // m^3/mol

    // Validate input
    assert(P >= P_eq && "Pressure must be greater than or equal to equilibrium pressure!");

    // Compute delta_G_v
    double delta_G_v = dV_ol_wd / V_initial * (P - P_eq);

    // Compute delta_G_hom
    double delta_G_hom = (16 * M_PI * std::pow(gamma, 3)) / (3 * std::pow(delta_G_v, 2));

    // Compute Q_a
    double Q_a = dH_a + P * V_star;

    // Compute nucleation rate
    double I = K0 * T * std::exp(-delta_G_hom / (k * T)) * std::exp(-Q_a / (R * T));
    return I;
}

// Function to calculate dimensionless time (sigma_s)
double calculate_sigma_s(double I_PT, double Y_PT, double d_0, double kappa, double D) {
    // Check for invalid inputs
    if (I_PT == 0 || Y_PT == 0 || d_0 == 0) {
        return std::numeric_limits<double>::infinity(); // Return infinity for invalid inputs
    }

    // Compute sigma_s using the given formula
    double sigma_s = (kappa / std::pow(D, 2)) * std::pow((I_PT * std::pow(Y_PT, 2) * d_0) / 6.7, -1.0 / 3.0);
    return sigma_s;
}

// Function to calculate dimensionless time (sigma_s) for vector inputs
double calculate_sigma_s(const std::vector<double>& I_array, const std::vector<double>& Y_array, double d_0, double kappa, double D) {
    // Validate input sizes
    if (I_array.size() != Y_array.size()) {
        throw std::invalid_argument("I_array and Y_array must have the same size");
    }

    // Compute the average of I_array and Y_array
    double I_PT = std::accumulate(I_array.begin(), I_array.end(), 0.0) / I_array.size();
    double Y_PT = std::accumulate(Y_array.begin(), Y_array.end(), 0.0) / Y_array.size();

    // Call the scalar version of calculate_sigma_s
    return calculate_sigma_s(I_PT, Y_PT, d_0, kappa, D);
}

// Function to calculate the Avrami number using corrected Equation (19)
double calculate_avrami_number_yoshioka_2015(double I_max, double Y_max, double kappa, double D) {
    // Compute the Avrami number
    return std::pow(D * D / kappa, 4) * I_max * std::pow(Y_max, 3);
}

// Function to solve for the extended volume after site saturation
double solve_extended_volume_post_saturation(double Y, double s, double kappa, double D, double d0) {
    // Calculate the extended volume based on the given parameters
    double X3 = (6.7 * std::pow(D, 2) / (d0 * kappa)) * Y * s;
    return X3;
}

// Overloaded version for vector inputs
std::vector<double> solve_extended_volume_post_saturation(const double Y, const std::vector<double>& s, double kappa, double D, double d0) {
    // Validate input sizes

    // Compute extended volume for each pair of Y and s
    std::vector<double> X3(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        X3[i] = solve_extended_volume_post_saturation(Y, s[i], kappa, D, d0);
    }

    return X3;
}

// Basic Ode and solvers
// Define the ODE system based on the modified Equation (18)
std::vector<double> ode_system(double s, const std::vector<double>& X, double Av,
                                const std::function<double(double)>& Y_prime,
                                const std::function<double(double)>& I_prime) {
    // Extract state variables
    double X0 = X[0];
    double X1 = X[1];
    double X2 = X[2];
    double X3 = X[3];

    // Avrami factor
    double Av_factor = std::pow(Av, 0.25);

    // Calculate the derivatives based on the Avrami equation
    double dX3 = Av_factor * (4.0 * M_PI * Y_prime(s) * X2);
    double dX2 = Av_factor * (2.0 * Y_prime(s) * X1);
    double dX1 = Av_factor * (Y_prime(s) * X0);
    double dX0 = Av_factor * I_prime(s);

    // Return the derivatives as a vector
    return {dX0, dX1, dX2, dX3};
}

// Solve the Modified Equation (18)
std::vector<std::vector<double>> solve_modified_equations_eq18(
    double Av,
    const std::function<double(double)>& Y_prime_func,
    const std::function<double(double)>& I_prime_func,
    const std::pair<double, double>& s_span,
    const std::vector<double>& X_ini,
    int n_span,
    bool debug) {
    
    // Debugging output
    if (debug) {
        std::cout << "X0 = [";
        for (const auto& x : X_ini) {
            std::cout << x << " ";
        }
        std::cout << "]" << std::endl;

        std::cout << "s_span = (" << s_span.first << ", " << s_span.second << ")" << std::endl;
        std::cout << "Av = " << Av << std::endl;
        std::cout << "Y_prime_func(s_span.first) = " << Y_prime_func(s_span.first) << ", "
                  << "Y_prime_func(s_span.second) = " << Y_prime_func(s_span.second) << std::endl;
        std::cout << "I_prime_func(s_span.first) = " << I_prime_func(s_span.first) << ", "
                  << "I_prime_func(s_span.second) = " << I_prime_func(s_span.second) << std::endl;
    }

    // Define the ODE system for IRK4Solver
    auto odes = [&](double s, const std::vector<double>& X) -> std::vector<double> {
        return ode_system(s, X, Av, Y_prime_func, I_prime_func);
    };

    // Time step size
    double h = (s_span.second - s_span.first) / (n_span-1);

    // Create the solver
    IRK4Solver solver;

    // Solve the ODE system
    auto [t_values, X_values] = solver.solve(odes, X_ini, s_span, h, debug);

    // Safeguard for very small values
    const double threshold = 1e-12; // Define a small threshold
    for (auto& row : X_values) {
        for (auto& value : row) {
            if (std::abs(value) < threshold) {
                value = 0.0; // Set to zero to avoid computational issues
            }
        }
    }

    // Combine results for return
    std::vector<std::vector<double>> solution(X_values.size(), std::vector<double>(X_ini.size(), 0.0));
    for (size_t i = 0; i < X_values.size(); ++i) {
        solution[i] = X_values[i];
    }

    return solution;
}

// Constructor
MO_KINETICS::MO_KINETICS() {
    PT_eq = {0.0, 0.0, 0.0}; // Initialize P, T, cl to 0.0
}

// Set the kinetics model
void MO_KINETICS::setKineticsModel(std::function<double(double, double, double, double)> Y_func,
                                   std::function<double(double, double, double)> I_func) {
    Y_func_ori = Y_func;
    I_func_ori = I_func;
}

// Fix the kinetics model
void MO_KINETICS::setKineticsFixed(double P, double T, double Coh) {
    assert(!PT_eq.empty() && "PT_eq must be set before calling setKineticsFixed!");
    double P_eq = computeEqP(T);
    Y_func = [this, P, T, P_eq, Coh](double t) { return Y_func_ori(P, T, P_eq, Coh); };
    I_func = [this, P, T, P_eq](double t) { return I_func_ori(P, T, P_eq); };
}

// Set phase transition equilibrium
void MO_KINETICS::setPTEq(double P0, double T0, double cl) {
    PT_eq = {P0, T0, cl}; // Update vector with new values
}

double MO_KINETICS::computeEqP(double T) {
    return PT_eq[0] + PT_eq[2] * (T - PT_eq[1]);
}

std::pair<std::vector<std::vector<double>>, std::vector<bool>> MO_KINETICS::solveModifiedEquation(
    const std::pair<double, double>& t_span, 
    const std::vector<double>& X_ini, 
    bool is_saturated, 
    int n_span, 
    bool debug) {

    // Ensure previous steps are valid
    assert(Y_func && I_func && "Kinetics functions must be set before solving!");

    // Compute scaling variables
    double I_max = std::max(1e-50, 6.0 * I_func(t_span.first) / d0);
    double Y_max = std::max(1e-50, Y_func(t_span.first));
    double Av = calculate_avrami_number_yoshioka_2015(I_max, Y_max, kappa, D);
    
    // Define non-dimensionalized versions of Y_func and I_func
    auto Y_prime_func = [this, Y_max](double s) {
        return Y_func(s * t_scale) / Y_max; // Scale time by t_scale
    };

    auto I_prime_func = [this, I_max](double s) {
        return 6.0 * I_func(s * t_scale) / d0 / I_max; // Scale time by t_scale
    };
    

    if (debug) {
        std::cout << "solveModifiedEquation: I_max = " << I_max << ", Y_max = " << Y_max << ", Av = " << Av << std::endl;
    }

    // Compute scaling factors
    X_scale_array = {
        std::pow(I_max, 0.75) * std::pow(Y_max, -0.75),
        std::pow(I_max, 0.5) * std::pow(Y_max, -0.5),
        std::pow(I_max, 0.25) * std::pow(Y_max, -0.25),
        1.0
    };

    // Non-dimensionalize the time span
    std::pair<double, double> s_span = {t_span.first / t_scale, t_span.second / t_scale};
    std::vector<double> s_values(n_span);
    for (int i = 0; i < n_span; ++i) {
        s_values[i] = s_span.first + i * (s_span.second - s_span.first) / (n_span - 1);
    }

    std::vector<double> I_array(n_span), Y_array(n_span);
    for (int i = 0; i < n_span; ++i) {
        I_array[i] = 6.0*I_func(s_values[i] * t_scale)/d0;
        Y_array[i] = Y_func(s_values[i] * t_scale);
    }

    // Compute saturation condition
    double s_saturation = calculate_sigma_s(I_array, Y_array, d0, kappa, D);

    if (debug) {
        std::cout << "solveModifiedEquation: t_saturation = " << s_saturation * t_scale << "\n";
        std::cout << "solveModifiedEquation: is_saturated = " << is_saturated << "\n";
    }

    // Initialize result containers
    std::vector<std::vector<double>> X_array(4, std::vector<double>(n_span, 0.0));
    std::vector<bool> is_saturated_array(n_span, false);

    if (!is_saturated) {
        auto it = std::find_if(s_values.begin(), s_values.end(), [s_saturation](double s) { return s > s_saturation; });
        if (it != s_values.end() && std::distance(s_values.begin(), it) > 1) {
            // Pre-saturation & saturation
            int i0 = std::distance(s_values.begin(), it);
            std::pair<double, double> s_span_us = {s_values[0], s_values[i0]};
            auto solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span_us, X_ini, i0, debug);


            // Scale and store pre-saturation results
            for (size_t j = 0; j < solution_nd.size(); ++j) {
                for (size_t k = 0; k < solution_nd[j].size(); ++k) {
                    X_array[k][j] = solution_nd[j][k] * X_scale_array[k];
                }
            }

            // Post-saturation
            auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
            for (int i = i0; i < n_span; ++i) {
                X_array[3][i] = post_saturation[i];
            }
            std::fill(is_saturated_array.begin() + i0, is_saturated_array.end(), true);
        }
        else if (it != s_values.end() && std::distance(s_values.begin(), it) <= 1) {
            // saturation at the 0th or 1st sub-step
            auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
            for (size_t i = 0; i < n_span; ++i) {
                X_array[3][i] = post_saturation[i];
            }
            
            int i0 = std::distance(s_values.begin(), it);
            std::fill(is_saturated_array.begin()+i0, is_saturated_array.end(), true);
        } else {
            // Full non-saturation
            auto solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span, X_ini, n_span, debug);
            
            // Debug: Print X_scale_array if debug is true
            if (debug) {
                std::cout << "X_scale_array: ";
                for (const auto& scale : X_scale_array) {
                    std::cout << scale << " ";
                }
                std::cout << "\n";
            }

            // std::cout << "solution_nd.size() = " << solution_nd.size() << std::endl;

            for (size_t j = 0; j < solution_nd.size(); ++j) {
                for (size_t k = 0; k < solution_nd[j].size(); ++k) {
                    // Debug print for each value
                    // std::cout << "solution_nd[" << j << "][" << k << "] = "  << solution_nd[j][k] << ", X_scale_array[" << k << "] = " << X_scale_array[k] << std::endl;
                    // X_array[j][k] = 0.0;
                    X_array[k][j] = solution_nd[j][k] * X_scale_array[k];
                }
            }
        }
    } else {
        // Full saturation
        auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
        for (size_t i = 0; i < n_span; ++i) {
            X_array[3][i] = post_saturation[i];
        }
        std::fill(is_saturated_array.begin(), is_saturated_array.end(), true);
    }

    return {X_array, is_saturated_array};
}


std::vector<std::vector<double>> MO_KINETICS::solve(double P, double T, double t_max, int n_t, int n_span, bool debug) {
    // Initialize variables
    std::vector<double> X = {0.0, 0.0, 0.0, 0.0};
    bool is_saturated = false;
    std::vector<std::vector<double>> results(n_t * n_span, std::vector<double>(n_col, 0.0));

    // Compute equilibrium pressure
    double Peq = computeEqP(T);

    // Loop over time steps
    for (int i_t = 0; i_t < n_t; ++i_t) {
        if (debug) {
            std::cout << "i_t: " << i_t << std::endl;
        }

        // Define the time span for the current step
        double t_piece_min = t_max / n_t * i_t;
        double t_piece_max = t_max / n_t * (i_t + 1);
        std::pair<double, double> t_span = {t_piece_min, t_piece_max};

        std::vector<std::vector<double>> X_array(4, std::vector<double>(n_span, 0.0));
        std::vector<bool> is_saturated_array(n_span, false);

        if (P > Peq) {
            // Solve the kinetics if equilibrium condition is met
            auto solution = solveModifiedEquation(t_span, X, is_saturated, n_span, debug);
            X_array = solution.first;
            is_saturated_array = solution.second;
            X = {X_array[0].back(), X_array[1].back(), X_array[2].back(), X_array[3].back()};
            is_saturated = is_saturated_array.back();
        } else {
            // Assign trivial values if equilibrium condition is not met
            X = {0.0, 0.0, 0.0, 0.0};
            is_saturated = false;
        }

        // Compute results for each step in the current time span
        std::vector<double> V_array(n_span);
        for (int j = 0; j < n_span; ++j) {
            double threshold = 50.0; // Define a threshold for large values
            if (X_array[3][j] > threshold) {
                // If X_array[3][j] is too large, directly set V_array[j] to 1
                V_array[j] = 1.0;
            } else {
                // Otherwise, compute the exponential term
                V_array[j] = 1.0 - std::exp(-X_array[3][j]);
            }

            results[i_t * n_span + j] = {
                t_piece_min + (t_piece_max - t_piece_min) / n_span * j,
                X_array[0][j],
                X_array[1][j],
                X_array[2][j],
                X_array[3][j],
                V_array[j],
                static_cast<double>(is_saturated_array[j])
            };
        }
    }

    return results;
}
