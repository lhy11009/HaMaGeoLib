#include "metastable.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <functional>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

struct TestCase {
    std::string name;
    std::function<void()> func;
};

void test_metastable_hosoya_06_eq2() {
    // This test correlates to the "test_growth_rate_scalar_inputs_bd" in test_metastable.py
    // Tests the grain growth rate.
    double P = 12366300000.0 + 1e9;    // Pressure in Pascals
    double T = 1173.15;   // Temperature in Kelvin
    double Coh = 150;  // Concentration in wt.ppm H2O
    
    // equilibrium conditions
    double PT_P = 13.5e9;     // Pa
    double PT_T = 1740.0;   // K
    double PT_cl = 2e6;     // Clapeyron slope

    // compute the equilibrium condition
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    //instantiate the class
    PTKinetics pTKinetics;

    // Expected value from Python
    double result_std = 5.77783763960406e-13;

    // Calculate result
    double result = pTKinetics.growth_rate(P, T, P_eq, T_eq, Coh);

    // Verify the result is within tolerance
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Metastable calculation failed");
}

void test_nucleation_rate_scalar_inputs() {
    // This test correlates to the "test_nulceaion_rate_scalar_inputs_bd" in test_metastable.py
    // Tests the nucleation rate
    double P = 12366300000.0 + 0.3e9;    // Pressure in Pascals
    double T = 1173.15;   // Temperature in Kelvin
    double Coh = 150;  // Concentration in wt.ppm H2O
    
    // equilibrium conditions
    double PT_P = 13.5e9;     // Pa
    double PT_T = 1740.0;   // K
    double PT_cl = 2e6;     // Clapeyron slope

    // compute the equilibrium condition
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    //instantiate the class
    PTKinetics pTKinetics;

    // Expected value from Python
    double result_std = 3.6648422297565646e-09;

    // Calculate result
    double result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq);

    // Verify the result is within tolerance
    if (std::abs(result - result_std) / result_std >= 1e-6)
    {
        std::cerr << "Nucleation rate calculation failed.\n"
                  << "Expected: " << result_std << ", "
                  << "Got: " << result << ", "
                  << "Relative error: " << std::abs(result - result_std) / result_std << std::endl;
        throw std::runtime_error("Nucleation rate number verification failed.");
    }
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Nucleation rate calculation failed!");
}

void test_nucleation_rate_scalar_inputs_big() {
    // This test correlates to the "test_nulceaion_rate_scalar_inputs_bd" in test_metastable.py
    double P = 14.75e9;    // Pressure in Pascals
    double T = 600 + 273.15 ;   // Temperature in Kelvin
    double Coh = 150;  // Concentration in wt.ppm H2O
    
    // equilibrium conditions
    double PT_P = 13.5e9;     // Pa
    double PT_T = 1740.0;   // K
    double PT_cl = 2e6;     // Clapeyron slope

    // compute the equilibrium condition
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    //instantiate the class
    PTKinetics pTKinetics;

    // Expected value from Python
    double result_std = 5.14297e+21;

    // Calculate result
    double result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq);

    // Verify the result is within tolerance
    if (std::abs(result - result_std) / result_std >= 1e-6)
    {
        std::cerr << "Nucleation rate calculation failed.\n"
                  << "Expected: " << result_std << ", "
                  << "Got: " << result << ", "
                  << "Relative error: " << std::abs(result - result_std) / result_std << std::endl;
        throw std::runtime_error("Nucleation rate number verification failed.");
    }
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Nucleation rate calculation failed!");
}

// todo_nu
void test_nucleation_rate_surface_scalar_inputs() {
    // This test correlate to the test "test_nucleation_rate_scalar_inputs" in test_metastable.py
    double P = 1.5e10;   // Pressure in Pascals
    double T = 1000;     // Temperature in Kelvin
    double P_eq = 1.4e10; // Equilibrium pressure in Pascals
    double T_eq = 1000;     // Equilirbium temperature in Kelvin
    
    //instantiate the class
    PTKinetics pTKinetics;

    // Expected result from Python
    double result_std = 903686.9915438003;

    // Calculate nucleation rate
    double result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq);

    // Verify the result is within tolerance
    if (std::abs(result - result_std) / result_std >= 1e-6)
    {
        std::cerr << "Nucleation rate calculation failed.\n"
                  << "Expected: " << result_std << ", "
                  << "Got: " << result << ", "
                  << "Relative error: " << std::abs(result - result_std) / result_std << std::endl;
        throw std::runtime_error("Nucleation rate number verification failed.");
    }
    assert(std::abs(result - result_std) / result_std < 1e-6 && "Nucleation rate calculation failed!");
}

void test_avrami_number() {
    // This is related to the "test_avrami_number" from the test_metastable.py
    // Test parameters
    double P = 1.5e10;       // Pressure in Pascals
    double T = 1000.0;       // Temperature in Kelvin
    double P_eq = 1.4e10;    // Equilibrium pressure in Pascals
    double T_eq = 1000.0;    // Equilibrium pressure in Pascals
    double d0 = 0.01;        // Parental grain size in meters
    double Coh = 1000.0;     // Cohesion in wt.ppm H2O

    //instantiate the class
    PTKinetics pTKinetics;
    
    // Compute I_max and Y_max
    double I_max = std::max(1e-50, pTKinetics.nucleation_rate(P, T, P_eq, T_eq));
    double Y_max = std::max(1e-50, pTKinetics.growth_rate(P, T, P_eq, T_eq, Coh));

    // Compute the Avrami number
    double Av = calculate_avrami_number_yoshioka_2015(I_max, Y_max);

    // Expected result
    double Av_expected = 1.2917411325502283e+32;

    // Check result against expected value
    if (std::abs(Av - Av_expected) / Av_expected >= 1e-6)
    {
        std::cerr << "Avrami number calculation failed.\n"
                  << "Expected: " << Av_expected << ", "
                  << "Got: " << Av << ", "
                  << "Relative error: " << std::abs(Av - Av_expected) / Av_expected << std::endl;
        throw std::runtime_error("Avrami number verification failed.");
    }
}

void test_solve_modified_equations_eq18() {
    // This is related to the "test_solve_modified_equations_eq18" in test_metastable.py
    // Tests utility of  solve_modified_equations_eq18
    // Set the condition
    double P = 12.5e9; // Pa
    double T = 623.75; // K
    double Coh = 1000.0;

    // Equilibrium condition
    double PT_P = 14e9;     // Pa
    double PT_T = 1760.0;   // K
    double PT_cl = 4e6;     // Clapeyron slope
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    // Scaling parameters
    double D = 100e3;       // m
    double d0 = 1e-2;       // m
    double kappa = 1e-6;    // m²/s
    double t_scale = std::pow(D, 2.0) / kappa;

    // Compute parameters
    //instantiate the class
    PTKinetics pTKinetics;
    
    double I0 =  pTKinetics.nucleation_rate(P, T, P_eq, T_eq); // per unit volume
    double Y0 =  pTKinetics.growth_rate(P, T, P_eq, T_eq, Coh);
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
    std::vector<double> expected_X1 = {0.348959, 0.121773, 0.044499, 0.015528};

    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());
}

void test_solve_modified_equations_eq18_1() {
    // This is related to the "test_solve_modified_equations_eq18_1" test in test_metastable.py
    // Similar to test_solve_modified_equations_eq18, but with a higher T.
    
    // Test parameters
    const double P = 14.75e9;       // Pressure (Pa)
    const double T = 600 + 273.15;  // Temperature (K)
    double Coh = 1000.0;

    // Equilibrium condition
    double PT_P = 14e9;     // Pa
    double PT_T = 1760.0;   // K
    double PT_cl = 4e6;     // Clapeyron slope
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    // Scaling parameters
    double D = 100e3;       // m
    double d0 = 1e-2;       // m
    double kappa = 1e-6;    // m²/s
    double t_scale = std::pow(D, 2.0) / kappa;

    // Compute parameters
    PTKinetics pTKinetics;
    double I0 =  pTKinetics.nucleation_rate(P, T, P_eq, T_eq); // per unit volume
    double Y0 =  pTKinetics.growth_rate(P, T, P_eq, T_eq, Coh);
    double Av = calculate_avrami_number_yoshioka_2015(I0, Y0);

    // Define Y_prime_func and I_prime_func
    auto Y_prime_func = [](double s) -> double { return 1.0; };
    auto I_prime_func = [](double s) -> double { return 1.0; };

    std::pair<double, double> s_span = {0.0, 3.1536e+12/t_scale};
    std::vector<double> X = {0.0, 0.0, 0.0, 0.0};

    int n_span = 10; // Number of time steps

    // Compute the solution
    std::vector<std::vector<double>> solution_nd = solve_modified_equations_eq18(
        Av, Y_prime_func, I_prime_func, s_span, X, n_span, false);

    // Extract the last solution
    std::vector<double> X1(solution_nd[solution_nd.size()-1]);

    // Expected result
    std::vector<double> expected_X1 = {9.68121176e+06, 9.37258611e+13, 9.50206018e+20, 9.19914568e+27};

    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());
}

void test_solve_modified_equations_eq18_1S() {
    // This is related to the "test_solve_modified_equations_eq18_1S" test in test_metastable.py
    
    // Test parameters
    const double P = 14.75e9;       // Pressure (Pa)
    const double T = 600 + 273.15;  // Temperature (K)
    double Coh = 1000.0;

    // Equilibrium condition
    double PT_P = 14e9;     // Pa
    double PT_T = 1760.0;   // K
    double PT_cl = 4e6;     // Clapeyron slope
    double P_eq = (T - PT_T) * PT_cl + PT_P;
    double T_eq =  (P - PT_P) / PT_cl + PT_T; // K

    // Scaling parameters
    double D = 100e3;       // m
    double d0 = 1e-2;       // m
    double kappa = 1e-6;    // m²/s
    double t_scale = std::pow(D, 2.0) / kappa;

    // Compute parameters

    const double A=exp(-18.0);
    const double n=3.2;
    const double dHa=274e3;
    const double V_star_growth=3.3e-6;
    const double gamma=0.6;
    const double fs=1e-3;
    const double K0=1e30;
    const double Vm=4.05e-5;
    const double dS=7.7;
    const double dV=3.16e-6;
    const double nucleation_type=1;

    PTKinetics pTKinetics(A, n, dS, dV, 
                          dHa, V_star_growth, fs, Vm,
                          gamma, K0, nucleation_type);

    double I0 =  pTKinetics.nucleation_rate(P, T, P_eq, T_eq); // per unit volume
    double Y0 =  pTKinetics.growth_rate(P, T, P_eq, T_eq, Coh);
    double Av = calculate_avrami_number_yoshioka_2015(I0, Y0);

    // Define Y_prime_func and I_prime_func
    auto Y_prime_func = [](double s) -> double { return 1.0; };
    auto I_prime_func = [](double s) -> double { return 1.0; };

    std::pair<double, double> s_span = {0.0, 3.1536e+12/t_scale};
    std::vector<double> X = {0.0, 0.0, 0.0, 0.0};

    int n_span = 10; // Number of time steps

    // Compute the solution
    std::vector<std::vector<double>> solution_nd = solve_modified_equations_eq18(
        Av, Y_prime_func, I_prime_func, s_span, X, n_span, false);

    // Extract the last solution
    std::vector<double> X1(solution_nd[solution_nd.size()-1]);

    // Expected result
    std::vector<double> expected_X1 = {7.00416718e+04,4.90583579e+09,3.59830629e+14,2.52031388e+19};

    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());
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
    // This is correlated with "test_solve_values" test in the test_metastable.py
    // Define constants
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 1000.0; // wt.ppm H2O

    // Phase transition parameters for equilibrium
    const double PT410_P = 13.5e9;   // Equilibrium pressure (Pa)
    const double PT410_T = 1740.0; // Equilibrium temperature (K)
    const double PT410_cl = 2e6;   // Clapeyron slope

    // Test parameters
    const double P = 14.75e9;       // Pressure (Pa)
    const double T = 600 + 273.15;  // Temperature (K)
    const double t_max = 10e6 * year; // Maximum time (s)
    const int n_t = 100;            // Number of time intervals
    const int n_span = 10;          // Number of steps within each interval

    // Initialize the MO_KINETICS class
    MO_KINETICS Mo_Kinetics;
    Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
    Mo_Kinetics.linkAndSetKineticsModel();
    Mo_Kinetics.setKineticsFixed(P, T, Coh);

    // Solve the kinetics
    auto results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, false);

    // Check the shape of the results
    assert(results.size() == static_cast<size_t>(n_t * n_span+1));
    assert(results[0].size() == 7);

    // Expected result for results[10]
    std::vector<double> expected_X1 = {
        3.15360000e+12,  // Time
        1.42038933e+25,  // N
        9.75531060e+13,  // Dn
        7.01622359e+02,  // S
        4.72894318e+00,  // Dimensionless volume
        9.91164196e-01 ,  // Volume fraction
        1.00000000e+00   // Saturation status
    };

    // Extract the actual row from results[10]
    auto X1 = results[10];

    // Verify
    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());
}

// test solve at a given P and T
void test_solve_values_lowT() {
    // This is related to the test "test_solve_values_low_T" from test_metastable.py
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
    Mo_Kinetics.linkAndSetKineticsModel();
    Mo_Kinetics.setKineticsFixed(P, T, Coh);

    // Solve the kinetics
    auto results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, false);

    // // Check the shape of the results
    assert(results.size() == static_cast<size_t>(n_t * n_span + 1));
    assert(results[0].size() == 7);

    // Expected result for results[10]
    std::vector<double> expected_X1 = {
        3.15360000e+12,  // Time
        0.00000000e+00,  // N
        0.00000000e+00,  // Dn
        0.00000000e+00,  // S
        0.00000000e+00,  // Dimensionless volume
        0.00000000e+00,  // Volume fraction
        0.00000000e+00   // Saturation status
    };

    // Extract the actual row from results[10]
    auto X1 = results[10];

    // Verify each element in the row matches the expected value
    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());
}

void test_solve_profile() {
    // This test is related to test_solve_profile in the python script
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 150.0; // wt.ppm H2O
    const double d0 = 1e-2; // grain size, m

    // Phase transition parameters for equilibrium
    const double PT410_P = 13.5e9;   // Equilibrium pressure (Pa)
    const double PT410_T = 1740.0; // Equilibrium temperature (K)
    const double PT410_cl = 2e6;   // Clapeyron slope

    // Test parameters
    const int n_t = 1;            // Number of time intervals
    const int n_span = 10;          // Number of steps within each interval

    // Initialize the MO_KINETICS class
    const double A=exp(-18.0);
    const double n=3.2;
    const double dHa=274e3;
    const double V_star_growth=3.3e-6;
    const double gamma=0.46;
    const double fs=6e-4;
    const double K0=1e30;
    const double Vm=4.05e-5;
    const double dS=7.7;
    const double dV=3.16e-6;
    const double nucleation_type=1;
    
    MO_KINETICS Mo_Kinetics(d0, A, n, dS, dV, dHa, V_star_growth, fs, Vm,
                                gamma, K0, nucleation_type);
    Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
    Mo_Kinetics.linkAndSetKineticsModel();

    // Load profile data
    fs::path current_file = __FILE__; // full path to this source file
    fs::path three_up = current_file.parent_path().parent_path().parent_path();
    fs::path ifile = three_up / "tests/fixtures/research/haoyuan_metastable_subduction/foo_contour_data.txt";

    std::ifstream infile(ifile);
    assert(infile && "Input file not found!");

    std::string line;
    std::getline(infile, line); // skip header

    std::vector<double> foo_contour_Ps;
    std::vector<double> foo_contour_Ts;
    std::vector<double> foo_contour_ts;

    // Load data
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        double P, T, t;
        if (iss >> P >> T >> t) {
            foo_contour_Ps.push_back(P);
            foo_contour_Ts.push_back(T);
            foo_contour_ts.push_back(t);
        }
    }

    // Set metastable contents
    std::vector<double> foo_contents_wl_mo(foo_contour_Ps.size(), 0.0);
    
    std::shared_ptr<std::vector<std::vector<double>>> p_results;
   
    for (size_t i = 0; i < foo_contour_Ps.size() - 1; ++i) {
        double P = foo_contour_Ps[i];
        double T = foo_contour_Ts[i];
        double t0 = foo_contour_ts[i];
        double t1 = foo_contour_ts[i + 1];

        Mo_Kinetics.setKineticsFixed(P, T, Coh); // set condition in kinetic
         
        if (i == 0){
            p_results  = std::make_shared<std::vector<std::vector<double>>>(
                            Mo_Kinetics.solve(P, T, t0, t1, n_t, n_span, false)
                        );
        }
        else{
            const std::vector<double>& last_vec = p_results->back();

            std::vector<double> X_ini(last_vec.begin()+1, last_vec.begin()+5);
            bool is_saturated = static_cast<bool>(last_vec[6]);
            

            p_results  = std::make_shared<std::vector<std::vector<double>>>(
                            Mo_Kinetics.solve(P, T, t0, t1, n_t, n_span, false, X_ini, is_saturated)
                        );
            
        }
        const std::vector<double>& last_vec = p_results->back();
        foo_contents_wl_mo[i+1] = last_vec[5];
        
        bool is_saturated = static_cast<bool>(last_vec[6]);
    }

    // Verify the result is within tolerance
    const double result_std = 0.010276906577713185;
    if (std::abs(foo_contents_wl_mo[237] - result_std) / result_std >= 1e-6)
    {
        std::cerr << "Transformed volume calculation failed.\n"
                  << "Expected: " << result_std << ", "
                  << "Got: " << foo_contents_wl_mo[237]  << ", "
                  << "Relative error: " << std::abs(foo_contents_wl_mo[237] - result_std) / result_std << std::endl;
        throw std::runtime_error("Transformed volume verification failed.");
    }
    assert(std::abs(foo_contents_wl_mo[237] - result_std) / result_std < 1e-6 && "Nucleation rate calculation failed!");
}

// test solve at a given P and T
void test_solve_values_test_aspect() {
    // This test is not related to python test, instead, it's used to benchmark values from aspect run time checks.
    // Define constants
    // outputs at line outputs 2017 in the test case
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 1000.0; // wt.ppm H2O

    // Phase transition parameters for equilibrium
    const double PT410_P = 14e9;   // Equilibrium pressure (Pa)
    const double PT410_T = 1760.0; // Equilibrium temperature (K)
    const double PT410_cl = 4e6;   // Clapeyron slope

    // Test parameters
    const double P = 2.66654e+10;       // Pressure (Pa)
    const double T = 917.658;  // Temperature (K)
    const double t_max = 1000.0 * year; // Maximum time (s)
    const int n_t = 1;            // Number of time intervals
    const int n_span = 10;          // Number of steps within each interval

    // Initialize the MO_KINETICS class
    MO_KINETICS Mo_Kinetics;
    Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
    Mo_Kinetics.linkAndSetKineticsModel();
    Mo_Kinetics.setKineticsFixed(P, T, Coh);

    // Solve the kinetics
    auto results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, false);

    // Check the shape of the results
    assert(results.size() == static_cast<size_t>(n_t * n_span + 1));
    assert(results[0].size() == 7);

    // Expected result for results[10]
    std::vector<double> expected_X1 = {
        t_max,  // Time
        2.7627787e25,  // N
        1.3605372955e14,  // Dn
        701.622359,  // S
        0.002627,  // Dimensionless volume
        0.002624,  // Volume fraction
        1.0   // Saturation status
    };

    // Extract the actual row from results[10]
    auto X1 = results[10];

    // Verify each element in the row matches the expected value
    bool pass = true;
    std::ostringstream oss;
    for (size_t i = 0; i < X1.size(); ++i) {
        double relative_error = std::abs(X1[i] - expected_X1[i]) / expected_X1[i];
        if (relative_error >= 1e-2) {
            oss << std::fixed << std::setprecision(6);
            oss << "Value mismatch at index " << i << ":\n"
                << "  Expected:  " << expected_X1[i] << "\n"
                << "  Computed:  " << X1[i] << "\n"
                << "  Rel. error: " << relative_error << " (threshold: 1e-2)\n";
            pass = false;
            }
    }
    if (!pass)
        throw std::runtime_error(oss.str());

}



int main() {
    std::vector<TestCase> tests = {
        {"test_metastable_hosoya_06_eq2", test_metastable_hosoya_06_eq2},
        {"test_nucleation_rate_scalar_inputs", test_nucleation_rate_scalar_inputs},
        {"test_nucleation_rate_scalar_inputs_big", test_nucleation_rate_scalar_inputs_big},
        {"test_avrami_number", test_avrami_number},
        {"test_solve_modified_equations_eq18", test_solve_modified_equations_eq18},
        {"test_solve_modified_equations_eq18_1", test_solve_modified_equations_eq18_1},
        {"test_solve_modified_equations_eq18_1S", test_solve_modified_equations_eq18_1S},
        {"test_solve_values", test_solve_values},
        {"test_solve_values_lowT", test_solve_values_lowT},
        {"test_solve_profile", test_solve_profile},
        {"test_solve_values_test_aspect", test_solve_values_test_aspect},
    };

    int pass_count = 0;
    for (const auto& test : tests) {
        try {
            test.func();
            std::cout << "[PASS] " << test.name << "\n";
            ++pass_count;
        } catch (const std::exception& e) {
            std::cerr << "[FAIL] " << test.name << ": " << e.what() << "\n";
        }
    }

    std::cout << "\nSummary: " << pass_count << "/" << tests.size() << " tests passed.\n";
    return (pass_count == tests.size()) ? 0 : 1;
}
