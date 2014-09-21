
#ifndef baxcat_cxx_datamodels_msd
#define baxcat_cxx_datamodels_msd

#include <vector>
#include <cmath>

#include "prng.hpp"
#include "utils.hpp"
#include "distributions/multinomial.hpp"
#include "distributions/symmetric_dirichlet.hpp"

using std::vector;

namespace baxcat{
namespace models{

    template <typename T>
    struct __MDAllowed__;


    // allow only unsigned integral types
    template <> struct __MDAllowed__<size_t>{};
    template <> struct __MDAllowed__<uint_fast8_t>{};
    template <> struct __MDAllowed__<uint_fast16_t>{};
    template <> struct __MDAllowed__<uint_fast32_t>{};


    template <typename T>
    struct MultinomialDirichlet : __MDAllowed__<T>{
        static void suffstatInsert(T x, std::vector<T> &counts)
        {
            baxcat::dist::multinomial::suffstatInsert(x, counts);
        }

        static void suffstatRemove(T x, std::vector<T> &counts)
        {
            baxcat::dist::multinomial::suffstatRemove(x, counts);
        }

        static double logPrior(const std::vector<double> &p, double alpha)
        {
            return baxcat::dist::symmetric_dirichlet::logPdf(p, alpha);
        }

        static double logZ( double n, const std::vector<T> &counts, double alpha )
        {
            double K = static_cast<double>( counts.size() );
            return log( n + alpha*K );
        }

        static double logLikelihood(const std::vector<T> &counts, const std::vector<double> &p)
        {
            return baxcat::dist::multinomial::logPdf(counts, p);
        }

        static double logMarginalLikelihood(double n, const std::vector<T> &counts, double alpha)
        {
            double A = static_cast<double>(counts.size())*alpha;
            double sum_lgamma = 0;
            double K = static_cast<double>(counts.size());
            for( auto w : counts)
                sum_lgamma += lgamma( static_cast<double>(w)+alpha );
            return lgamma(A) - lgamma(A+n) + sum_lgamma - K*lgamma(alpha);
        }

        // todo: cache logZ
        static double logPredictiveProbability(T x, const std::vector<T> &counts, double alpha,
                                               double log_z)
        {
            double counts_w = static_cast<double>(counts[x]);
            double K = static_cast<double>(counts.size());
            // return log( alpha + counts_w ) - log_z;
            return log( alpha + counts_w ) - log(baxcat::utils::sum(counts)+alpha*K);
        }

        static double logSingletonProbability(T x, double alpha, double log_z){
            return log( alpha ) - log_z;
        }

        static size_t predictiveSample(const std::vector<T> &counts, double alpha,
                                       baxcat::PRNG *rng, double log_z)
        {
            vector<double> logp(counts.size(),0);
            for(size_t k = 0; k < counts.size(); ++k)
                logp[k] =  logPredictiveProbability(k, counts, alpha, log_z);
            return rng->lpflip(logp);
        }

        // TODO: Implement
        static double posteriorSample(){
            return -1;
        }
    };

}}

#endif
