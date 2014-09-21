
#ifndef baxcat_cxx_distributions_students_t
#define baxcat_cxx_distributions_students_t

#include <vector>
#include <random>
#include <cmath>
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace students_t{


static double logPdf(double x, double df)
{
    return lgamma((df+1)/2)-lgamma(df/2)-.5*log(df*M_PI)-((df+1)/2)*log(1+x*x/df);
}


static double logPdf(std::vector<double> X, double df)
{
    double logp = 0;
    for (auto &x : X)
        logp += logPdf(x, df);

    return logp;
}


}}}

#endif
