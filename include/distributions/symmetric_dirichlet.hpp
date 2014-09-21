
#ifndef baxcat_cxx_distributions_symmetric_dirichlet
#define baxcat_cxx_distributions_symmetric_dirichlet

#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace symmetric_dirichlet{


static double logPdf(std::vector<double> X, double alpha)
{
    size_t K = X.size();
    double A = 0;
    double alpha_minus_one = alpha-1;
    for (auto &x : X)
        A += alpha_minus_one*log(x);

    return A+K*lgamma(alpha)-lgamma(alpha*K);
}


}}}

#endif
