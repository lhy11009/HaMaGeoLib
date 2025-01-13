#include "IRK4Solver.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// Wave equation function
std::vector<double> wave_equation(double t, const std::vector<double>& y, double c, double dx) {
    size_t N = y.size() / 2;
    std::vector<double> u(y.begin(), y.begin() + N);
    std::vector<double> v(y.begin() + N, y.end());

    std::vector<double> dudt = v;
    std::vector<double> dvdt(N, 0.0);

    // Finite difference for spatial derivatives
    for (size_t i = 1; i < N - 1; ++i) {
        dvdt[i] = c * c * (u[i - 1] - 2.0 * u[i] + u[i + 1]) / (dx * dx);
    }

    // Combine dudt and dvdt into a single vector
    std::vector<double> dydt(2 * N, 0.0);
    std::copy(dudt.begin(), dudt.end(), dydt.begin());
    std::copy(dvdt.begin(), dvdt.end(), dydt.begin() + N);

    return dydt;
}

int main() {
    // Problem setup
    double L = 10.0;  // Length of the domain
    size_t N = 100;   // Number of spatial points
    double dx = L / (N - 1);  // Spatial step size
    double c = 1.0;   // Wave speed
    double h = 0.01;  // Time step
    std::pair<double, double> t_span = {0.0, 5.0};

    // Initial conditions
    std::vector<double> x(N);
    for (size_t i = 0; i < N; ++i) {
        x[i] = i * dx;
    }

    std::vector<double> u0(N);
    for (size_t i = 0; i < N; ++i) {
        u0[i] = std::exp(-0.5 * std::pow(x[i] - L / 2, 2));
    }
    std::vector<double> v0(N, 0.0);

    std::vector<double> y0(2 * N);
    std::copy(u0.begin(), u0.end(), y0.begin());
    std::copy(v0.begin(), v0.end(), y0.begin() + N);

    // Create solver
    IRK4Solver solver;

    // Solve the wave equation
    auto f = [&](double t, const std::vector<double>& y) -> std::vector<double> {
        return wave_equation(t, y, c, dx);
    };

    auto [t_values, y_values] = solver.solve(f, y0, t_span, h);

    // Validate specific value at t = 450 * h and x = x[10]
    size_t time_index = 450;  // Corresponds to t = 450 * h
    size_t spatial_index = 10;  // x[10]
    double expected_value = 0.2806255370282081;
    double tolerance = 1e-6;

    double u_value = y_values[time_index][spatial_index];
    assert(std::abs(u_value - expected_value) < tolerance);

    std::cout << "Test passed: u_values[450, 10] = " << u_value << " (Expected: " << expected_value << ")" << std::endl;
    return 0;
}
