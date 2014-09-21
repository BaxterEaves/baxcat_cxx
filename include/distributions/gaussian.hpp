
#ifndef baxcat_cxx_distributions_gaussian
#define baxcat_cxx_distributions_gaussian

#include <vector>
#include <cmath>
#include "numerics.hpp"

namespace baxcat {
namespace dist {
namespace gaussian{


static void suffstatInsert(double x, double &sum_x, double &sum_x_sq)
{
    sum_x += x;
    sum_x_sq += x*x;
}


static void suffstatRemove(double x, double &sum_x, double &sum_x_sq)
{
    sum_x -= x;
    sum_x_sq -= x*x;
}


static double logPdfSuffstats(double n, double sum_x, double sum_x_sq, double mu, double rho)
{
    return  -.5*n * LOG_2PI + .5*n * log(rho) -.5*(rho*(n*mu*mu - 2*mu*sum_x + sum_x_sq));
}


static double logPdf(double x, double mu, double rho)
{
    return logPdfSuffstats(1, x, x*x, mu, rho);
}


static double logPdf(std::vector<double> X, double mu, double rho)
{
    double n = 0;
    double sum_x = 0;
    double sum_x_sq = 0;
    for (auto &x : X){
        ++n;
        suffstatInsert(x, sum_x, sum_x_sq);
    }
    return logPdfSuffstats(n, sum_x, sum_x_sq, mu, rho);
}


static double cdf(double x, double mu, double rho)
{
    return .5*(1+erf((x-mu)/ sqrt(2/rho)));
}


}}}

#endif
