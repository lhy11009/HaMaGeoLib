#include "metastable.h"
#include <cassert>
#include "IRK4Solver.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>  // For std::invalid_argument

// Constructor
PTKinetics::PTKinetics():
      A(std::exp(-18.0)), n(3.2),
      dS(7.7), dV(3.16e-6),
      dHa(274e3), Vstar(3.3e-6),
      fs(1e-3), Vm(4.05e-5),
      gamma(0.6), K0(3.65e38),
      nucleation_type(0)
{}

PTKinetics::PTKinetics(const double A_, const double n_, const double dS_, const double dV_, 
                       const double dHa_, const double Vstar_, const double fs_, const double Vm_
                       , const double gamma_, const double K0_, const int nucleation_type)
    : A(A_), n(n_), dS(dS_), dV(dV_),
    dHa(dHa_), Vstar(Vstar_), fs(fs_), Vm(Vm_),
    gamma(gamma_), K0(K0_), nucleation_type(nucleation_type)
{}

// Growth rate (without ΔGr term)
double PTKinetics::growth_rate_P1(double P, double T, double Coh) const {
    return A * std::pow(Coh, n) * std::exp(-(dHa + P * Vstar) / (R * T));
}

// Full growth rate (with ΔGr term)
double PTKinetics::growth_rate(double P, double T, double Peq, double Teq, double Coh) const {
    assert(P > Peq && "P must be greater than Peq");
    double dGr = dV * (P - Peq) - dS * (T - Teq);
    return growth_rate_P1(P, T, Coh) * T * (1.0 - std::exp(-dGr / (R * T)));
}

// Nucleation rate
double PTKinetics::nucleation_rate(double P, double T, double Peq, double Teq) const {
    assert(P > Peq && "P must be >= Peq");
    double dGr = dV * (P - Peq) - dS * (T - Teq);
    double delta_G_hom = (16.0 * M_PI * fs * std::pow(Vm, 2) * std::pow(gamma, 3)) / (3.0 * std::pow(dGr, 2));
    double Q_a = dHa + P * Vstar;
    return K0 * T * std::exp(-delta_G_hom / (k * T)) * std::exp(-Q_a / (R * T));
}

// Function to calculate the Avrami number using corrected Equation (19)
double calculate_avrami_number(double I_max, double Y_max, double kappa, double D) {
    // Compute the Avrami number
    return std::pow(D * D / kappa, 4) * I_max * std::pow(Y_max, 3);
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
    double dX3 = Av_factor * (4.0 * Y_prime(s) * X2);
    double dX2 = Av_factor * (M_PI * Y_prime(s) * X1);
    double dX1 = Av_factor * (2.0 * Y_prime(s) * X0);
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
        std::cout << "solve_modified_equations_eq18" << std::endl << "X0 = [";
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
    double h = (s_span.second - s_span.first) / n_span;

    // Create the solver
    IRK4Solver solver;

    // Solve the ODE system
    auto [t_values, X_values] = solver.solve(odes, X_ini, s_span, h, debug);

    if (debug){
        std::cout << "      h = " << h << std::endl;
        std::cout << "      X_values.size() = " << X_values.size() << std::endl;
        size_t _size = t_values.size();
        size_t _size1 = X_values[0].size();
        for (size_t i = 0; i < _size; ++i) {
            std::cout << "      t[" << i << "] = " << t_values[i] << std::endl;
            for (size_t j = 0; j < _size1; ++j) {
                std::cout << "      X["<< i << "][" << j << "] = " << X_values[i][j] << std::endl;
            }
        }
    }

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

// todo_metastable
// Constructor
MO_KINETICS::MO_KINETICS():
      d0(1e-2),
      A(std::exp(-18.0)), n(3.2),
      dS(7.7), dV(3.16e-6),
      dHa(274e3), Vstar(3.3e-6),
      fs(1e-3), Vm(4.05e-5),
      gamma(0.6), K0(3.65e38),
      nucleation_type(0), include_derivative(false)
{
    PT_eq = {0.0, 0.0, 0.0}; // Initialize P, T, cl to 0.0
    n_col = 7;
}

MO_KINETICS::MO_KINETICS(const double d0_, const double A_, const double n_, const double dS_, const double dV_, 
                       const double dHa_, const double Vstar_, const double fs_, const double Vm_
                       , const double gamma_, const double K0_, const int nucleation_type_, const bool include_derivative_)
    :d0(d0_), A(A_), n(n_), dS(dS_), dV(dV_),
    dHa(dHa_), Vstar(Vstar_), fs(fs_), Vm(Vm_),
    gamma(gamma_), K0(K0_), nucleation_type(nucleation_type_),
    include_derivative(include_derivative_)
{
    n_col = (this->include_derivative)? 8: 7;
}

void MO_KINETICS::linkAndSetKineticsModel() {
    kinetics = std::make_shared<PTKinetics>(
        A, n,
        dS, dV,             // dS, dv
        dHa, Vstar,                // dHa, Vstar
        fs, Vm,               //fs, Vm
        gamma, K0,            // gamma, K0
        nucleation_type // nucleation_type
    );
    assert(kinetics->get_nucleation_type() == nucleation_type);
}

// Fix the kinetics model
void MO_KINETICS::setKineticsFixed(double P, double T, double Coh) {
    assert(!PT_eq.empty() && "PT_eq must be set before calling setKineticsFixed!");

    double P_eq = computeEqP(T); // compute equilibrium values
    double T_eq = computeEqT(P);

    // fix grain growth and nucleation functions to specific P, T condition
    // convert nucleation rate to volumetric nucleation rate
    if (kinetics->get_nucleation_type() == 0){
        f_nu = 1.0;
    }
    else if (kinetics->get_nucleation_type() == 1){
        f_nu = 6.7 / d0;
    }
    else{
        throw std::runtime_error("This nucleation type is not implemented yet.");
    }
    Y_func = [this, P, T, P_eq, T_eq, Coh](double t) { return kinetics->growth_rate(P, T, P_eq, T_eq, Coh); };
    I_func = [this, P, T, P_eq, T_eq](double t) { return f_nu*kinetics->nucleation_rate(P, T, P_eq, T_eq); };
}

// Set phase transition equilibrium
void MO_KINETICS::setPTEq(double P0, double T0, double cl) {
    PT_eq = {P0, T0, cl}; // Update vector with new values
}

double MO_KINETICS::computeEqP(double T) {
    return PT_eq[0] + PT_eq[2] * (T - PT_eq[1]);
}

double MO_KINETICS::computeEqT(double P) {
    return PT_eq[1] + (P - PT_eq[0]) / PT_eq[2];
}

double MO_KINETICS::growth_rate(double P, double T, double Coh) {
    double P_eq = computeEqP(T); // compute equilibrium values
    double T_eq = computeEqT(P);
    
    // return 0 if equilibrium phase transition condition is not met
    if (P > P_eq){
        return kinetics->growth_rate(P, T, P_eq, T_eq, Coh);
    }
    else{
        return 0.0;
    }
}

double MO_KINETICS::nucleation_rate(double P, double T) {
    double P_eq = computeEqP(T); // compute equilibrium values
    double T_eq = computeEqT(P);

    // return 0 if equilibrium phase transition condition is not met
    if (P > P_eq){
        return f_nu*kinetics->nucleation_rate(P, T, P_eq, T_eq);
    }
    else{
        return 0.0;
    }
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
    double I_max = std::max(1e-50, I_func(t_span.first));
    double Y_max = std::max(1e-50, Y_func(t_span.first));
    double Av = calculate_avrami_number(I_max, Y_max, kappa, D);
    
    // Define non-dimensionalized versions of Y_func and I_func
    auto Y_prime_func = [this, Y_max](double s) {
        return Y_func(s * t_scale) / Y_max; // Scale time by t_scale
    };

    auto I_prime_func = [this, I_max](double s) {
        return I_func(s * t_scale) / I_max; // Scale time by t_scale
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
   
    // Non-dimensionalize the initial solution
    std::vector<double> X_ini_nd(4, 0.0);
    for (size_t k = 0; k < X_ini.size(); ++k) {
        X_ini_nd[k] = X_ini[k] / X_scale_array[k];
    }

    // Non-dimensionalize the time span
    std::pair<double, double> s_span = {t_span.first / t_scale, t_span.second / t_scale};
    std::vector<double> s_values(n_span+1);
    for (int i = 0; i <= n_span; ++i) {
        s_values[i] = s_span.first + i * (s_span.second - s_span.first) / n_span;
    }

    std::vector<double> I_array(n_span+1), Y_array(n_span+1);
    for (int i = 0; i <= n_span; ++i) {
        I_array[i] = I_func(s_values[i] * t_scale);
        Y_array[i] = Y_func(s_values[i] * t_scale);
    }

    // Compute saturation condition
    double s_saturation = calculate_sigma_s(I_array, Y_array, d0, kappa, D);

    if (debug) {
        std::cout << "I_array[0] = " << I_array[0] << std::endl;
        std::cout << "Y_array[0] = " << Y_array[0] << std::endl;
        std::cout << "solveModifiedEquation: s_saturation = " << s_saturation << "\n";
        std::cout << "solveModifiedEquation: t_saturation = " << s_saturation * t_scale << "\n";
        std::cout << "solveModifiedEquation: is_saturated = " << is_saturated << "\n";
    }

    // Initialize result containers
    std::vector<std::vector<double>> X_array(4, std::vector<double>(n_span+1, 0.0));
    std::vector<bool> is_saturated_array(n_span+1, false);

    if (!is_saturated) {
        auto it = std::find_if(s_values.begin(), s_values.end(), [s_saturation, s_values](double s) { return s > s_values[0] + s_saturation; });
        if (it != s_values.end() && std::distance(s_values.begin(), it) > 1) {
            // Pre-saturation & saturation
            int i0 = std::distance(s_values.begin(), it);
            std::pair<double, double> s_span_us = {s_values[0], s_values[i0]};
            auto solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span_us, X_ini_nd, i0, debug);


            // Scale and store pre-saturation results
            for (size_t j = 0; j < solution_nd.size(); ++j) {
                for (size_t k = 0; k < solution_nd[j].size(); ++k) {
                    X_array[k][j] = solution_nd[j][k] * X_scale_array[k];
                }
            }
            // Pre-assign values post-saturation
            for (size_t j = solution_nd.size(); j < static_cast<size_t>(n_span + 1); ++j) {
                for (size_t k = 0; k < solution_nd[solution_nd.size()-1].size(); ++k) {
                    X_array[k][j] = solution_nd[solution_nd.size()-1][k] * X_scale_array[k];
                }
            }

            // Post-saturation, increment from values derived by the analytical solution
            auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
            auto post_saturation_ini = solve_extended_volume_post_saturation(Y_max, s_values[i0], kappa, D, d0);
            for (int i = i0; i <= n_span; ++i) {
                X_array[3][i] = X_array[3][i0] + post_saturation[i] - post_saturation_ini;
            }
            std::fill(is_saturated_array.begin() + i0, is_saturated_array.end(), true);
        }
        else if (it != s_values.end() && std::distance(s_values.begin(), it) <= 1) {
            // assign the initial values
            for (size_t k = 0; k < X_ini.size(); ++k) {
                X_array[k][0] = X_ini[k];
            }
            
            // solve the nucleation in range of s_values[0], s_saturation
            std::pair<double, double> s_span_us = {s_values[0], s_values[0] + s_saturation};
            
            auto solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span_us, X_ini_nd, n_span, debug);

            for (size_t j = 1; j < static_cast<size_t>(n_span + 1); ++j) {
                for (size_t k = 0; k < solution_nd[j].size(); ++k) {
                    X_array[k][j] = solution_nd[solution_nd.size()-1][k] * X_scale_array[k];
                }
            }
            
            // saturation at the 1st sub-step
            // Post-saturation, increment from values derived by the analytical solution
            auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
            auto post_saturation_ini = solve_extended_volume_post_saturation(Y_max, s_values[1], kappa, D, d0);
            for (size_t j = 1; j < static_cast<size_t>(n_span + 1); ++j) {
                X_array[3][j] = X_array[3][1] + post_saturation[j] - post_saturation_ini;
            }
            
            std::fill(is_saturated_array.begin()+1, is_saturated_array.end(), true);
        } else {
            // solve the unsaturated condition
            auto solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span, X_ini_nd, n_span, debug);
            
            // Debug: Print X_scale_array if debug is true
            if (debug) {
                std::cout << "X_scale_array: ";
                for (const auto& scale : X_scale_array) {
                    std::cout << scale << " ";
                }
                std::cout << "\n";
            }

            // assign the values from the solution
            for (size_t j = 0; j < solution_nd.size(); ++j) {
                for (size_t k = 0; k < solution_nd[j].size(); ++k) {
                    // Debug print for each value
                    if (debug){
                        std::cout << "solution_nd[" << j << "][" << k << "] = "  << solution_nd[j][k] << ", X_scale_array[" << k << "] = " << X_scale_array[k] << std::endl;
                    }
                    X_array[k][j] = solution_nd[j][k] * X_scale_array[k];
                }
            }
        }
    } else {
        // assign the initial values
        for (size_t j = 0; j < static_cast<size_t>(n_span + 1); ++j) {
            for (size_t k = 0; k < X_ini.size(); ++k) {
                X_array[k][j] = X_ini[k];
            }
        }

        // Full saturation, increment from values derived by the analytical solution
        auto post_saturation = solve_extended_volume_post_saturation(Y_max, s_values, kappa, D, d0);
        auto post_saturation_ini = solve_extended_volume_post_saturation(Y_max, s_values[0], kappa, D, d0);
        for (size_t j = 0; j < static_cast<size_t>(n_span + 1); ++j) {
            X_array[3][j] = X_ini[3] + post_saturation[j] - post_saturation_ini;
        }
        std::fill(is_saturated_array.begin(), is_saturated_array.end(), true);
    }

    return {X_array, is_saturated_array};
}

std::vector<std::vector<double>> MO_KINETICS::solve(double P, double T, double t_min, double t_max, int n_t, int n_span, bool debug, 
    std::vector<double> X, bool is_saturated) {
    
    // Initialize variables
    // Check if X has exactly 4 elements
    if (X.size() != 4) {
        throw std::invalid_argument("Error: X must have exactly 4 elements.");
    }
    
    std::vector<std::vector<double>> results(n_t * n_span+1, std::vector<double>(n_col, 0.0));

    // Compute equilibrium pressure
    double Peq = computeEqP(T);

    // todo_metastable
    double last_derivative = 0.0;
    // Loop over time steps
    for (int i_t = 0; i_t < n_t; ++i_t) {
        if (debug) {
            std::cout << "i_t: " << i_t << std::endl;
        }

        // Define the time span for the current step
        double t_piece_min = t_min + (t_max - t_min) / n_t * i_t;
        double t_piece_max = t_min + (t_max - t_min) / n_t * (i_t + 1);
        std::pair<double, double> t_span = {t_piece_min, t_piece_max};

        std::vector<std::vector<double>> X_array(4, std::vector<double>(n_span+1, 0.0));
        std::vector<bool> is_saturated_array(n_span+1, false);

        if (P > Peq) {
            // Solve the kinetics if equilibrium condition is met
            auto solution = solveModifiedEquation(t_span, X, is_saturated, n_span, false);
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
        std::vector<double> V_array(n_span+1);
        const double t_interval = (t_piece_max - t_piece_min) / n_span;
        for (int j = 0; j <= n_span; ++j) {
            double threshold = 50.0; // Define a threshold for large values
            if (X_array[3][j] > threshold) {
                // If X_array[3][j] is too large, directly set V_array[j] to 1
                V_array[j] = 1.0;
            } else {
                // Otherwise, compute the exponential term
                V_array[j] = 1.0 - std::exp(-X_array[3][j]);
            }

            // note i_t * n_span + j would have the same value
            // for the last node the piece and the first node in the consequtive piece
            // compute the derivative and recode the last value to append the next "first" value
            const double derivative = (j==0)? last_derivative: (V_array[j] - V_array[j-1])/t_interval;
            if (j == n_span)
                last_derivative = derivative;

            if (this->include_derivative)
            {
                results[i_t * n_span + j] = {
                    t_piece_min + t_interval * j,
                    X_array[0][j],
                    X_array[1][j],
                    X_array[2][j],
                    X_array[3][j],
                    V_array[j],
                    static_cast<double>(is_saturated_array[j]),
                    derivative
                };
            }
            else{
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
    }

    return results;
}
