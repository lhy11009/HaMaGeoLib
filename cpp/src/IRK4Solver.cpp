#include "IRK4Solver.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

// Constructor
IRK4Solver::IRK4Solver() {
    double sqrt3 = std::sqrt(3.0) / 6.0;

    A = {
        {0.25, 0.25 - sqrt3},
        {0.25 + sqrt3, 0.25}
    };

    b = {0.5, 0.5};
    c = {0.5 - sqrt3, 0.5 + sqrt3};
}

// Solve method with debug option
std::pair<std::vector<double>, std::vector<std::vector<double>>> 
IRK4Solver::solve(const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
                  const std::vector<double>& y0, const std::pair<double, double>& t_span, double h, bool debug) {
    double t0 = t_span.first;
    double t_end = t_span.second;

    std::vector<double> t_values = {t0};
    std::vector<std::vector<double>> y_values = {y0};

    std::vector<double> y = y0;
    double t = t0;

    while (t < t_end) {
        if (t + h > t_end) {
            h = t_end - t;  // Adjust step size for the last step
        }

        size_t n = y.size();
        size_t stages = b.size();

        // Residual function for nonlinear solver
        auto residual = [&](const std::vector<std::vector<double>>& Y) -> std::vector<std::vector<double>> {
            std::vector<std::vector<double>> res(stages, std::vector<double>(n, 0.0));
            for (size_t i = 0; i < stages; ++i) {
                for (size_t k = 0; k < n; ++k) {
                    res[i][k] = Y[i][k] - y[k];
                    for (size_t j = 0; j < stages; ++j) {
                        auto f_eval = f(t + c[j] * h, Y[j]);
                        res[i][k] -= h * A[i][j] * f_eval[k];
                    }
                }
            }
            return res;
        };

        // Initial guess for stages
        std::vector<std::vector<double>> Y(stages, y);

        // Solve for Y using simple fixed-point iteration (replaceable with a better solver)
        size_t max_iterations = 100;
        double tolerance = 1e-6;
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            auto res = residual(Y);

            // Debug: Print residuals, relative residuals, and temporary solutions if debug is true
            if (debug) {
                std::cout << "Iteration " << iter << ", t = " << t << ":\n";
                for (size_t i = 0; i < stages; ++i) {
                    std::cout << "  Stage " << i << " Residual: ";
                    for (size_t k = 0; k < n; ++k) {
                        std::cout << res[i][k] << " ";
                    }
                    std::cout << "\n";

                    std::cout << "  Stage " << i << " Relative Residual: ";
                    for (size_t k = 0; k < n; ++k) {
                        double relative_residual = std::abs(res[i][k]) / (std::abs(Y[i][k]) + 1e-12);
                        std::cout << relative_residual << " ";
                    }
                    std::cout << "\n";

                    std::cout << "  Stage " << i << " Solution: ";
                    for (size_t k = 0; k < n; ++k) {
                        std::cout << Y[i][k] << " ";
                    }
                    std::cout << "\n";
                }
            }

            // Update Y values based on residuals
            bool converged = true;
            for (size_t i = 0; i < stages; ++i) {
                for (size_t k = 0; k < n; ++k) {
                    double relative_residual = std::abs(res[i][k]) / (std::abs(Y[i][k]) + 1e-12); // Add small value to avoid division by zero
                    Y[i][k] -= res[i][k];
                    if (relative_residual > tolerance) {
                        converged = false;
                    }
                }
            }

            if (converged) break;
            if (iter == max_iterations - 1) {
                throw std::runtime_error("Nonlinear solver failed to converge");
            }
        }

        // Update solution using stages
        std::vector<double> dy(n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            for (size_t i = 0; i < stages; ++i) {
                dy[k] += b[i] * f(t + c[i] * h, Y[i])[k];
            }
        }

        for (size_t k = 0; k < n; ++k) {
            y[k] += h * dy[k];
        }

        t += h;
        t_values.push_back(t);
        y_values.push_back(y);
    }

    return {t_values, y_values};
}
