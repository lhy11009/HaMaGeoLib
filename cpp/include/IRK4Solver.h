#ifndef IRK4SOLVER_H
#define IRK4SOLVER_H

#include <vector>
#include <functional>

class IRK4Solver {
public:
    IRK4Solver();
    
    std::pair<std::vector<double>, std::vector<std::vector<double>>> 
    solve(const std::function<std::vector<double>(double, const std::vector<double>&)>& f,
          const std::vector<double>& y0, const std::pair<double, double>& t_span, double h, bool debug=false);

private:
    std::vector<std::vector<double>> A;  // IRK4 A coefficients
    std::vector<double> b;               // IRK4 b coefficients
    std::vector<double> c;               // IRK4 c coefficients
};

#endif // IRK4SOLVER_H
