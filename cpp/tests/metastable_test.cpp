#include "metastable.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <algorithm>

void test_metastable_hosoya_06_eq2() {
    double P = 1e7;    // Pressure in Pascals
    double T = 1000;   // Temperature in Kelvin
    double P_eq = 9e6; // Equilibrium pressure in Pascals
    double Coh = 100;  // Concentration in wt.ppm H2O

    // Expected value from Python
    double result_std = 5.3613444537140843e-17;

    // Calculate result
    double result = metastable_hosoya_06_eq2(P, T, P_eq, Coh);

    // Verify the result is within tolerance
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Metastable calculation failed");
}

void test_nucleation_rate_scalar_inputs() {
    double P = 1.5e10;   // Pressure in Pascals
    double T = 1000;     // Temperature in Kelvin
    double P_eq = 1.4e10; // Equilibrium pressure in Pascals

    // Expected result from Python
    double result_std = 596390.1390563988;

    // Calculate nucleation rate
    double result = nucleation_rate_yoshioka_2015(P, T, P_eq);

    // Verify the result is within tolerance
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Nucleation rate calculation failed!");
}

void test_avrami_number() {
    // Test parameters
    double P = 1.5e10;       // Pressure in Pascals
    double T = 1000.0;       // Temperature in Kelvin
    double P_eq = 1.4e10;    // Equilibrium pressure in Pascals
    double d0 = 0.01;        // Parental grain size in meters
    double Coh = 1000.0;     // Cohesion in wt.ppm H2O

    // Compute I_max and Y_max
    double I_max = std::max(1e-50, 6.0 * nucleation_rate_yoshioka_2015(P, T, P_eq) / d0);
    double Y_max = std::max(1e-50, metastable_hosoya_06_eq2(P, T, P_eq, Coh));

    // Compute the Avrami number
    double Av = calculate_avrami_number_yoshioka_2015(I_max, Y_max);

    // Expected result
    double Av_expected = 2.550621254633427e+34;

    // Check result against expected value
    assert(std::abs(Av - Av_expected) / Av_expected < 1e-6 &&
           "Avrami number calculation failed.");

    std::cout << "Test passed: Avrami number matches the expected value.\n";
}

void test_solve_modified_equations_eq18() {
    // Set the condition
    double P = 12.5e9; // Pa
    double T = 623.75; // K
    double Coh = 1000.0;

    // Equilibrium condition
    double PT_P = 14e9;     // Pa
    double PT_T = 1760.0;   // K
    double PT_cl = 4e6;     // Clapeyron slope
    double P_eq = (T - PT_T) * PT_cl + PT_P;

    // Scaling parameters
    double D = 100e3;       // m
    double d0 = 1e-2;       // m
    double kappa = 1e-6;    // m²/s
    double t_scale = std::pow(D, 2.0) / kappa;

    // Compute parameters
    double I0 = 6.0 * nucleation_rate_yoshioka_2015(P, T, P_eq) / d0; // per unit volume
    double Y0 = metastable_hosoya_06_eq2(P, T, P_eq, Coh);
    double Av = calculate_avrami_number_yoshioka_2015(I0, Y0);

    // Define Y_prime_func and I_prime_func
    auto Y_prime_func = [](double s) -> double { return 1.0; };
    auto I_prime_func = [](double s) -> double { return 1.0; };

    std::pair<double, double> s_span = {0.0, 3.1536e+12 / t_scale};
    std::vector<double> X = {0.0, 0.0, 0.0, 0.0};

    int n_span = 10; // Number of time steps

    // Compute the solution
    std::vector<std::vector<double>> solution_nd = solve_modified_equations_eq18(
        Av, Y_prime_func, I_prime_func, s_span, X, n_span, false);

    // Extract the last solution
    std::vector<double> X1(solution_nd[solution_nd.size()-1]);

    // Expected result
    std::vector<double> expected_X1 = {1.54409765e-02, 1.19211878e-04, 1.22716521e-06, 5.95288741e-08};

    bool test_passed = true;

    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            test_passed = false;
            std::cerr << "Mismatch detected at index " << i << ":\n";
            std::cerr << "  Calculated value: " << X1[i] << "\n";
            std::cerr << "  Expected value: " << expected_X1[i] << "\n";
            std::cerr << "  Relative error: " << relative_error << "\n";
        }
    }

    if (test_passed) {
        std::cout << "Test passed: solve_modified_equations_eq18 produces the expected results.\n";
    } else {
        std::cerr << "Test failed: Mismatch detected in the results.\n";
    }
}

void test_solve_modified_equations_eq18_1() {
    
    // Test parameters
    const double P = 14.75e9;       // Pressure (Pa)
    const double T = 600 + 273.15;  // Temperature (K)
    double Coh = 1000.0;

    // Equilibrium condition
    double PT_P = 14e9;     // Pa
    double PT_T = 1760.0;   // K
    double PT_cl = 4e6;     // Clapeyron slope
    double P_eq = (T - PT_T) * PT_cl + PT_P;

    // Scaling parameters
    double D = 100e3;       // m
    double d0 = 1e-2;       // m
    double kappa = 1e-6;    // m²/s
    double t_scale = std::pow(D, 2.0) / kappa;

    // Compute parameters
    double I0 = 6.0 * nucleation_rate_yoshioka_2015(P, T, P_eq) / d0; // per unit volume
    double Y0 = metastable_hosoya_06_eq2(P, T, P_eq, Coh);
    double Av = calculate_avrami_number_yoshioka_2015(I0, Y0);

    // Define Y_prime_func and I_prime_func
    auto Y_prime_func = [](double s) -> double { return 1.0; };
    auto I_prime_func = [](double s) -> double { return 1.0; };

    std::pair<double, double> s_span = {0.0, 0.00031536};
    std::vector<double> X = {0.0, 0.0, 0.0, 0.0};

    int n_span = 10; // Number of time steps

    // Compute the solution
    std::vector<std::vector<double>> solution_nd = solve_modified_equations_eq18(
        Av, Y_prime_func, I_prime_func, s_span, X, n_span, false);

    // Extract the last solution
    std::vector<double> X1(solution_nd[solution_nd.size()-1]);

    // Expected result
    std::vector<double> expected_X1 = {1.93416120e+06, 1.87048978e+12, 2.41188584e+18, 1.46554544e+25};

    bool test_passed = true;

    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            test_passed = false;
            std::cerr << "Mismatch detected at index " << i << ":\n";
            std::cerr << "  Calculated value: " << X1[i] << "\n";
            std::cerr << "  Expected value: " << expected_X1[i] << "\n";
            std::cerr << "  Relative error: " << relative_error << "\n";
        }
    }

    if (test_passed) {
        std::cout << "Test passed: solve_modified_equations_eq18_1 produces the expected results.\n";
    } else {
        std::cerr << "Test failed: Mismatch detected in the results.\n";
    }
}

// Utility function to check if two values are close
bool isClose(double a, double b, double rtol = 1e-6) {
    if (std::abs(b) < 1e-12) {  // Handle the case where b is close to or exactly zero
        return std::abs(a) < rtol; // Check if a is within the tolerance
    }
    return std::abs(a - b) / std::abs(b) < rtol;
}

// test solve at a given P and T
void test_solve_values() {
    // Define constants
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 1000.0; // wt.ppm H2O

    // Phase transition parameters for equilibrium
    const double PT410_P = 14e9;   // Equilibrium pressure (Pa)
    const double PT410_T = 1760.0; // Equilibrium temperature (K)
    const double PT410_cl = 4e6;   // Clapeyron slope

    // Test parameters
    const double P = 14.75e9;       // Pressure (Pa)
    const double T = 600 + 273.15;  // Temperature (K)
    const double t_max = 10e6 * year; // Maximum time (s)
    const int n_t = 100;            // Number of time intervals
    const int n_span = 10;          // Number of steps within each interval

    // Initialize the MO_KINETICS class
    MO_KINETICS Mo_Kinetics;
    Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
    Mo_Kinetics.setKineticsModel(metastable_hosoya_06_eq2, nucleation_rate_yoshioka_2015);
    Mo_Kinetics.setKineticsFixed(P, T, Coh);

    // Solve the kinetics
    auto results = Mo_Kinetics.solve(P, T, t_max, n_t, n_span, false);

    // Check the shape of the results
    assert(results.size() == static_cast<size_t>(n_t * n_span));
    assert(results[0].size() == 7);

    // Expected result for results[10]
    std::vector<double> expected_row = {
        3.15360000e+12,  // Time
        0.00000000e+00,  // N
        0.00000000e+00,  // Dn
        0.00000000e+00,  // S
        4.22192710e+00,  // Dimensionless volume
        9.85329654e-01,  // Volume fraction
        1.00000000e+00   // Saturation status
    };

    // Extract the actual row from results[10]
    auto actual_row = results[10];

    // Verify each element in the row matches the expected value
    for (size_t i = 0; i < expected_row.size(); ++i) {
        if (!isClose(actual_row[i], expected_row[i])) {
            std::cerr << "Mismatch in row 10 at index " << i
                      << ": expected " << expected_row[i]
                      << ", got " << actual_row[i] << std::endl;
            assert(false && "Row 10 mismatch");
        }
    }

    std::cout << "Test passed: solve() produces the expected results.\n";
}

// test solve at a given P and T
void test_solve_values_lowT() {
    // Define constants
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 1000.0; // wt.ppm H2O

    // Phase transition parameters for equilibrium
    const double PT410_P = 14e9;   // Equilibrium pressure (Pa)
    const double PT410_T = 1760.0; // Equilibrium temperature (K)
    const double PT410_cl = 4e6;   // Clapeyron slope

    // Test parameters
    const double P = 9e9;       // Pressure (Pa)
    const double T = 273.15;  // Temperature (K)
    const double t_max = 10e6 * year; // Maximum time (s)
    const int n_t = 100;            // Number of time intervals
    const int n_span = 10;          // Number of steps within each interval

    // Initialize the MO_KINETICS class
    MO_KINETICS Mo_Kinetics;
    Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
    Mo_Kinetics.setKineticsModel(metastable_hosoya_06_eq2, nucleation_rate_yoshioka_2015);
    Mo_Kinetics.setKineticsFixed(P, T, Coh);

    // Solve the kinetics
    auto results = Mo_Kinetics.solve(P, T, t_max, n_t, n_span, false);

    // Check the shape of the results
    assert(results.size() == static_cast<size_t>(n_t * n_span));
    assert(results[0].size() == 7);

    // Expected result for results[10]
    std::vector<double> expected_row = {
        3.15360000e+12,  // Time
        0.00000000e+00,  // N
        0.00000000e+00,  // Dn
        0.00000000e+00,  // S
        0.00000000e+00,  // Dimensionless volume
        0.00000000e+00,  // Volume fraction
        0.00000000e+00   // Saturation status
    };

    // Extract the actual row from results[10]
    auto actual_row = results[10];

    // Verify each element in the row matches the expected value
    for (size_t i = 0; i < expected_row.size(); ++i) {
        if (!isClose(actual_row[i], expected_row[i])) {
            std::cerr << "Mismatch in row 10 at index " << i
                      << ": expected " << expected_row[i]
                      << ", got " << actual_row[i] << std::endl;
            assert(false && "Row 10 mismatch");
        }
    }

    std::cout << "Test passed: solve() produces the expected results.\n";
}


int main() {
    test_metastable_hosoya_06_eq2();
    test_nucleation_rate_scalar_inputs();
    test_avrami_number();
    test_solve_modified_equations_eq18();
    test_solve_modified_equations_eq18_1();
    test_solve_values();
    test_solve_values_lowT();
    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}
