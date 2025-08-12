#ifndef METASTABLE_H
#define METASTABLE_H

#include <cmath>
#include <vector>
#include <functional>
#include <map>
#include <string>
#include <stdexcept>
#include <memory>
#include "IRK4Solver.h"

// class for kinetics
class PTKinetics {
public:
    // default constructor
    PTKinetics();

    // explicit constructor
    PTKinetics(const double A_, const double n_, const double dS_, const double dV_, 
                       const double dHa_, const double Vstar_, const double fs_, const double Vm_
                       , const double gamma_, const double K0_, const int nucleation_type);

    double growth_rate_P1(double P, double T, double Coh) const;
    double growth_rate(double P, double T, double Peq, double Teq, double Coh) const;
    double nucleation_rate(double P, double T, double Peq, double Teq) const;
    
    inline int get_nucleation_type() const{
        return nucleation_type;
    };

private:
    // Physical constants
    double Vm;

    // Growth parameters (Hosoya et al., 2006)
    double A;
    double n;
    double dHa;
    double Vstar;

    // Nucleation parameters (Yoshioka et al., 2015)
    double fs;
    double gamma;
    double K0;
    double dS;
    double dV;
    int nucleation_type;

    // type of nucleation: 0 - volumetric; 1 - surface; 2 - line; 3 - corner
    const double R = 8.31446; // gas constant
    const double k = 1.38e-23; // boltzman constant
};

// Function prototypes

double calculate_avrami_number(double I_max, double Y_max, double kappa = 1e-6, double D = 100e3);

double calculate_sigma_s(double I_PT, double Y_PT, double d_0, double kappa = 1e-6, double D = 100e3);
double calculate_sigma_s(const std::vector<double>& I_array, const std::vector<double>& Y_array, double d_0, double kappa = 1e-6, double D = 100e3);

// solution after cite situation
// The first function gives analytical solution
// The second function applys the solution by time increment
// Every function is overloaded with inputs of a double number or a vector
double solve_extended_volume_post_saturation(double Y, double s, double kappa = 1e-6, double D = 100e3, double d0 = 1e-2);
std::vector<double> solve_extended_volume_post_saturation(const double Y, const std::vector<double>& s, double kappa = 1e-6, double D = 100e3, double d0 = 1e-2);

std::vector<double> ode_system(double s, const std::vector<double>& X, double Av,
                                const std::function<double(double)>& Y_prime,
                                const std::function<double(double)>& I_prime); 

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
    // implicit constructor
    MO_KINETICS();

    // explicit constructor
    MO_KINETICS(const double d0_, const double A_, const double n_, const double dS_, const double dV_, 
                       const double dHa_, const double Vstar_, const double fs_, const double Vm_
                       ,const double gamma_, const double K0_, const int nucleation_type_, const bool include_derivative_);

    void linkAndSetKineticsModel();
    void setKineticsFixed(double P, double T, double Coh);
    void setPTEq(double P0, double T0, double cl);

    std::pair<std::vector<std::vector<double>>, std::vector<bool>>
    solveModifiedEquation(const std::pair<double, double>& t_span, const std::vector<double>& X_ini, 
                         bool is_saturated, int n_span = 10, bool debug = false);

    
    std::vector<std::vector<double>> 
    solve(const double P, const double T, const double t_min, const double t_max, const int n_t,
        const int n_span, const bool debug = false, const std::vector<double> X_ini = {0.0, 0.0, 0.0, 0.0},
        const bool is_saturated_ini = false);

    class MO_INITIATION_Error : public std::runtime_error {
    public:
        explicit MO_INITIATION_Error(const std::string& message) : std::runtime_error(message) {}
    };
   
    // functions to computer intermediate values
    double growth_rate(double P, double T, double Coh);
    
    double nucleation_rate(double P, double T);
    
    // utility functions
    double computeEqP(double T);
    
    double computeEqT(double P);

protected:
    double f_nu; // coefficent to the nucleation rate

private:

    // functions for solving the ODEs
    std::function<double(double, double, double, double, double)> Y_func_ori;
    std::function<double(double, double, double, double)> I_func_ori;
    std::function<double(double)> Y_func;
    std::function<double(double)> I_func;
    std::function<double(double)> Y_prime_func;
    std::function<double(double)> I_prime_func;
   

    // Number of result columns
    int n_col;     

    // free variables
    // Physical constants
    double Vm;
    // Growth parameters (Hosoya et al., 2006)
    double A;
    double n;
    double dHa;
    double Vstar;
    // Nucleation parameters (Yoshioka et al., 2015)
    double fs;
    double gamma;
    double K0;
    double dS;
    double dV;
    int nucleation_type;
    double d0 = 1e-2;    // Parental grain size
   
    // include derivative of volume
    bool include_derivative;
    
    // const variables
    const double kappa = 1e-6; // Thermal diffusivity
    const double D = 100e3;    // Slab thickness
    const double t_scale = D*D/kappa;

    // vector that set the equilirbium PT
    std::vector<double> PT_eq;
    
    // vector that saves the solution
    std::vector<double> X_scale_array;

    // pointer to an instantiation of the PTKinetics class
    std::shared_ptr<PTKinetics> kinetics;
};

#endif // METASTABLE_H
