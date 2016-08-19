
#ifndef baxcat_cxx_samplers_slice_guard
#define baxcat_cxx_samplers_slice_guard

#include "debug.hpp"

#include <limits>
#include <functional>

#include "prng.hpp"

namespace baxcat{

    struct Domain
    {
        double lower;
        double upper;
        Domain(double l, double u){
            lower = l;
            upper = u;
        }

        Domain(std::initializer_list<double> range){
            lower = *range.begin();
            upper = *(range.end()-1);
        }

        bool contains(double x){
            return x > lower and x < upper;
        }
    };

namespace samplers{

    template <typename lambda>
    double mhSample(double x_0, lambda &log_pdf, baxcat::Domain D,
            double w, size_t burn, baxcat::PRNG *rng)
    {
        double lp = log_pdf(x_0);

        for (size_t i=0; i < burn; ++i){
            double x_p = rng->normrand(x_0, w);
            if (D.contains(x_p)){
                double lp_prime = log_pdf(x_p);
                if (log(rng->rand()) < lp_prime-lp){
                    lp = lp_prime;
                    x_0 = x_p;
                }
            }
        }
        return x_0;
    };

    static double priormh(std::function<double(double)> loglike,
                          std::function<double()> draw, size_t burn,
                          baxcat::PRNG *rng)
    {
        auto x = draw();
        auto l = loglike(x);
        for (size_t i=0; i < burn; ++i){
            auto xp = draw();
            auto lp = loglike(xp);
            
            if (log(rng->rand()) < lp-l){
                l = lp;
                x = xp;
            }
        }

        return x;
    };

    template <typename lambda>
    void __stepout(double x_0, double y, double w, double m, lambda f,
            baxcat::Domain D, baxcat::PRNG *rng, double &L, double &R)
    {

        double U = rng->rand();
        L = x_0 - w*U;
        L = (L < D.lower) ? D.lower : L;
        R = L + w;

        double V = rng->rand();
        size_t J = floor(m*V);
        size_t K = (m-1)-J;

        while (J > 0 and y < f(L)) {
            L -= w;
            --J;
            if (L < D.lower)
                L = D.lower;
        }

        while (K > 0 and y < f(R)) {
            R += w;
            --K;
            if (R > D.upper)
                R = D.upper;
        }
    }

    // slice sample the log function log_pdf with bounds defined in domain, D,
    // and starting from point x_0. w is the expected slice width and burn the
    // the number of samples to ignore before the samples is collected.
    template <typename lambda>
    double sliceSample(double x_0, lambda &log_pdf, baxcat::Domain D, double w,
            size_t burn, baxcat::PRNG *rng)
    {
        const double m = 256;

        // interval
        double L,R;
        double y;
        size_t MAX_LOOPS = 50;

        bool reduce_width = false;

        reduce_width_and_try_again:
        if(reduce_width)
            w *= .5;

        for (size_t i = 0; i < burn; ++i) {
            y = log(rng->rand()) + log_pdf(x_0);
            __stepout(x_0, y, w, m, log_pdf, D, rng, L, R);
            size_t num_loops = 0;
            while(true){
                double U = rng->rand();
                double x_1 = L + U*(R-L);
                if( x_1 < D.lower)
                    x_1 = D.lower;
                double fx_1 = log_pdf(x_1);
                if(y < fx_1){
                    x_0 = x_1;
                    break;
                }

                if (x_1 < x_0)
                    L = ( x_1 < D.lower) ? D.lower : x_1;
                else
                    R = ( x_1 > D.upper) ? D.upper : x_1;

                ++num_loops;
                if( num_loops > MAX_LOOPS ){
                    reduce_width = true;
                    goto reduce_width_and_try_again; // yep. that's a goto
                    DEBUG_MESSAGE(std::cout, "Reached MAX_LOOPS");
                }
            }
        }

        return x_0;
    }

}}

#endif
