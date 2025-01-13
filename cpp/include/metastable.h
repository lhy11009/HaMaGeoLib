#ifndef METASTABLE_H
#define METASTABLE_H

#include <cmath>
#include <vector>
#include <functional>
#include <map>
#include <string>
#include <stdexcept>
#include "IRK4Solver.h"

// Function prototypes
double metastable_hosoya_06_eq2_P1(double P, double T, double Coh);
double metastable_hosoya_06_eq2(double P, double T, double P_eq, double Coh);

double nucleation_rate_yoshioka_2015(double P, double T, double P_eq);

double calculate_avrami_number_yoshioka_2015(double I_max, double Y_max, double kappa = 1e-6, double D = 100e3);

double calculate_sigma_s(double I_PT, double Y_PT, double d_0, double kappa = 1e-6, double D = 100e3);
double calculate_sigma_s(const std::vector<double>& I_array, const std::vector<double>& Y_array, double d_0, double kappa = 1e-6, double D = 100e3);

double solve_extended_volume_post_saturation(double Y, double s, double kappa = 1e-6, double D = 100e3, double d0 = 1e-2);
std::vector<double> solve_extended_volume_post_saturation(const double Y, const std::vector<double>& s, double kappa = 1e-6, double D = 100e3, double d0 = 1e-2);

// Solve the Modified Equation (18)
std::vector<std::vector<double>> solve_modified_equations_eq18(
    double Av,
    const std::function<double(double)>& Y_prime_func,
    const std::function<double(double)>& I_prime_func,
    const std::pair<double, double>& s_span,
    const std::vector<double>& X_ini,
    int n_span = 100,
    bool debug = false);

// Class for MO kinetics
class MO_KINETICS {
public:
    MO_KINETICS();

    void setKineticsModel(std::function<double(double, double, double, double)> Y_func_ori,
                          std::function<double(double, double, double)> I_func_ori);
    void setKineticsFixed(double P, double T, double Coh);
    void setPTEq(double P0, double T0, double cl);

    std::pair<std::vector<std::vector<double>>, std::vector<bool>>
    solveModifiedEquation(const std::pair<double, double>& t_span, const std::vector<double>& X_ini, 
                         bool is_saturated, int n_span = 10, bool debug = false);

    std::vector<std::vector<double>> 
    solve(double P, double T, double t_max, int n_t, int n_span, bool debug = false);

    class MO_INITIATION_Error : public std::runtime_error {
    public:
        explicit MO_INITIATION_Error(const std::string& message) : std::runtime_error(message) {}
    };

private:
    std::function<double(double, double, double, double)> Y_func_ori;
    std::function<double(double, double, double)> I_func_ori;
    std::function<double(double)> Y_func;
    std::function<double(double)> I_func;

    double kappa = 1e-6; // Thermal diffusivity
    double D = 100e3;    // Slab thickness
    double d0 = 1e-2;    // Parental grain size
    double t_scale = D*D/kappa;
    int n_col = 7;     // Number of result columns

    std::vector<double> X_scale_array;

    std::function<double(double)> Y_prime_func;
    std::function<double(double)> I_prime_func;

    std::vector<double> PT_eq;

    double computeEqP(double T);
};

#endif // METASTABLE_H
