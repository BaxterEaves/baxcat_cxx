
#ifndef baxcat_cxx_distributions_multinomial
#define baxcat_cxx_distributions_multinomial

#include <vector>
#include <cmath>
#include "template_helpers.hpp"
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace multinomial{


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> suffstatInsert(T x,
    std::vector<T> &counts)
{
    ++counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> suffstatRemove(T x,
std::vector<T> &counts)
{
    --counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdfSuffstats(std::vector<T> counts,
                                                                              std::vector<double> p)
{
    double logp = 0;
    for (size_t k = 0; k < counts.size(); ++k)
        logp += counts[k]*log(p[k]);

    return logp;
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdf(
    const std::vector<T> &X, std::vector<double> p)
{
    std::vector<T> counts(p.size(),0);
    for(auto &x : X)
        suffstatInsert(x, counts);
    return logPdfSuffstats(counts, p);
}

template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdf(T x, std::vector<double> p)
{
    return log(p[x]);
}


}}}

#endif
