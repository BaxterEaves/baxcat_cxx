
#ifndef baxcat_cxx_distributions_categorical
#define baxcat_cxx_distributions_categorical

#include <vector>
#include <cmath>
#include "template_helpers.hpp"
#include "utils.hpp"
#include "numerics.hpp"

namespace baxcat{
namespace dist{
namespace categorical{


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,void> suffstatInsert(T x,
    std::vector<T> &counts)
{
    ++counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,void> suffstatRemove(T x,
std::vector<T> &counts)
{
    --counts[x];
}


template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdfSuffstats(double n,
    std::vector<T> counts, std::vector<double> p)
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

    double n = static_cast<double>(baxcat::utils::sum(counts));
    return logPdfSuffstats(n, counts, p);
}

template <typename T>
static typename baxcat::enable_if<std::is_integral<T>,double> logPdf(T x, std::vector<double> p)
{
    std::vector<size_t> counts(p.size(),0);
    counts[x] = 1;
    return logPdfSuffstats(1, counts, p);
}


}}}

#endif
