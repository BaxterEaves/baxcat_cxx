
#ifndef baxcat_cxx_distributions_gamma
#define baxcat_cxx_distributions_gamma

#include <boost/math/distributions/gamma.hpp>
#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat {
namespace dist {
namespace gamma{


static double logPdf(double x, double shape, double scale)
{
    return -lgamma(shape)-shape*log(scale) + (shape-1)*log(x)-(x/scale);
}


static double logPdf(std::vector<double> X, double shape, double scale)
{
    double logp = 0;
    for (auto &x : X)
        logp += logPdf(x, shape, scale);

    return logp;
}


static double cdf(double x, double shape, double scale)
{
    boost::math::gamma_distribution<> dist(shape, scale);
    return cdf(dist, x);
}


}}}

#endif
