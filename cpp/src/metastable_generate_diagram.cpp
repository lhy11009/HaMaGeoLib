#include <iostream>
#include <fstream>
#include <filesystem> // For std::filesystem::path
#include <vector>
#include <cmath>
#include <cassert>
#include <iomanip> // For std::setprecision
#include "metastable.h"          // Include the MO_KINETICS class header
#include "IRK4Solver.h"          // Include the IRK4Solver class header
    
// Phase transition parameters for equilibrium
const double PT410_P = 14e9;   // Equilibrium pressure (Pa)
const double PT410_T = 1760.0; // Equilibrium temperature (K)
const double PT410_cl = 4e6;   // Clapeyron slope

void generate_results_diagram(const std::string& output_file) {
    // Define constants
    const double year = 365.0 * 24.0 * 3600.0; // Seconds in one year
    const double Coh = 1000.0;                 // wt.ppm H2O

    // Test parameters
    const double t_max = 10e6 * year; // Maximum time (s)
    const int n_t = 100;              // Number of time intervals
    const int n_span = 10;            // Number of steps within each interval

    // Ranges for P and T
    const double P_min = 0.0;
    const double P_max = 30e9;
    const double T_min = 273.15;
    const double T_max = 1873.15;
    const int P_steps = 200; // Number of steps for P
    const int T_steps = 100; // Number of steps for T

    // Step sizes
    const double dP = (P_max - P_min) / P_steps;
    const double dT = (T_max - T_min) / T_steps;

    // Total number of iterations
    const int total_iterations = (P_steps + 1) * (T_steps + 1);

    // Open the output file
    std::ofstream outfile(output_file);
    if (!outfile.is_open()) {
        std::cerr << "Error: Unable to open output file.\n";
        return;
    }

    // Header for the output file
    outfile << "P,T,Time,N,Dn,S,DimensionlessVolume,VolumeFraction,SaturationStatus\n";

    // Loop over the range of P and T
    int current_iteration = 0; // Counter for the current iteration
    for (int iP = 0; iP <= P_steps; ++iP) {
        for (int iT = 0; iT <= T_steps; ++iT) {
            // Print progress
            ++current_iteration;

            double P = P_min + iP * dP;
            double T = T_min + iT * dT;
            
            // Print progress, including P and T
            std::cout << "Iteration: " << current_iteration << " / " << total_iterations
                  << " | P: " << P << " Pa, T: " << T << " K\n";

            // Initialize the MO_KINETICS class
            MO_KINETICS Mo_Kinetics;
            Mo_Kinetics.setPTEq(PT410_P, PT410_T, PT410_cl);
            Mo_Kinetics.linkAndSetKineticsModel();
            Mo_Kinetics.setKineticsFixed(P, T, Coh);

            // Solve the kinetics
            bool debug = false;
            if (current_iteration == 126){
              debug = true;
            }
            auto results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug);

            // Write results to the file
            for (const auto& row : results) {
                outfile << std::fixed << std::setprecision(8); // Set precision for floating-point numbers
                outfile << P << "," << T; // Write P and T
                for (const auto& value : row) {
                    outfile << "," << value; // Write row values
                }
                outfile << "\n"; // End the row
            }
        }
    }

    // Close the output file
    outfile.close();
    std::cout << "Results saved to " << output_file << "\n";
}

int main(){

    // Determine the output file path relative to the current file
    std::filesystem::path current_file = __FILE__;
    std::filesystem::path output_file = current_file.parent_path() / "../../dtemp/metastable_diagram_cpp.txt";

    generate_results_diagram(output_file);
    return 0;
}