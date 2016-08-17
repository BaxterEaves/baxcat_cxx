
#ifndef baxcat_cxx_distributions_inverse_gamma
#define baxcat_cxx_distributions_inverse_gamma

#include <boost/math/distributions/inverse_gamma.hpp>
#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat {
namespace dist {
namespace inverse_gamma{


static double logPdf(double x, double shape, double scale)
{
    return shape*log(scale) - lgamma(shape) + (-shape-1)*log(x) - (scale/x);
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
    boost::math::inverse_gamma_distribution<> dist(shape, scale);
    return cdf(dist, x);
}


}}}

#endif
