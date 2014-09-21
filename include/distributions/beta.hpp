
#ifndef baxcat_cxx_distributions_beta
#define baxcat_cxx_distributions_beta

#include <cmath>
#include <vector>
#include <boost/math/distributions/beta.hpp>

#include "numerics.hpp"

namespace baxcat {
namespace dist {
namespace beta {


static void suffstatInsert(double x, double &sum_log_x, double &sum_log_minus_x)
{
    sum_log_x += log(x);
    sum_log_minus_x += log(1-x);
}


static void suffstatRemove(double x, double &sum_log_x, double &sum_log_minus_x)
{
    sum_log_x -= log(x);
    sum_log_minus_x -= log(1-x);
}


static double logPdfSuffstats(double n, double sum_log_x, double sum_log_minus_x, double a,
                              double b)
{
    return (a-1)*sum_log_x + (b-1)*sum_log_minus_x - n*baxcat::numerics::lbeta(a, b);
}


static double logPdf(double x, double a, double b)
{
    return logPdfSuffstats(1.0, log(x), log(1-x), a, b);
}


static double logPdf(std::vector<double> X, double a, double b)
{
    double sum_log_x = 0;
    double sum_log_minus_x = 0;
    double n = double(X.size());
    for( auto &x : X)
        suffstatInsert(x, sum_log_x, sum_log_minus_x);

    return logPdfSuffstats(n, sum_log_x, sum_log_minus_x, a, b);
}


static double cdf(double x, double a, double b)
{
    boost::math::beta_distribution<> dist(a, b);
    return cdf(dist, x);
}


}}}

#endif
