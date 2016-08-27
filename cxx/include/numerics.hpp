
#ifndef baxcat_cxx_distributions_numerics_hpp
#define baxcat_cxx_distributions_numerics_hpp

#include "debug.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cfloat>
#include <vector>

#define TOL 10e-8
#define ALMOST_ZERO DBL_MIN
#define LOG_2PI log(2.0*M_PI)
#define LOG_2 log(2.0)
#define LOG_PI log(M_PI)
#define INF std::numeric_limits<double>::infinity()
// #define NAN std::numeric_limits<double>::quiet_NaN()

namespace baxcat {
namespace numerics {

static double rgamma(double x, double a)
{
    double MAX_ITERS = 1000;
    double err = 0;
    double max_err = 10E-10;
    double Z = exp(a*log(x)-x-lgamma(a+1));

    double P_a = 0;
    double P_b = 0;

    double n = 0;

    do{
        P_a = P_b;
        P_b = P_a + exp(n*log(x)+lgamma(a+1)-lgamma(n+a+1));
        n++;
        err = fabs(P_a-P_b);
    }while(err > max_err && n < MAX_ITERS);

    ASSERT(std::cout, n > 1);
    ASSERT(std::cout, err <= max_err);

    if(n >= MAX_ITERS)
        throw MaxIterationsReached(static_cast<size_t>(MAX_ITERS));

    return Z*P_b;
}


template <typename T>
static int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}


template<typename T1, typename T2>
static double lbeta(T1 alpha, T2 beta)
{
    ASSERT_GREATER_THAN_ZERO(std::cout, alpha);
    ASSERT_GREATER_THAN_ZERO(std::cout, beta);
    return lgamma(alpha)+lgamma(beta)-lgamma(alpha+beta);
}


template<typename T>
static double lfactorial(T N)
{
    ASSERT(std::cout, N >= 0);
    return lgamma(N+1);
}


template<typename T1, typename T2>
static double lnchoosek(T1 n, T2 k)
{
    ASSERT(std::cout, n >=0);
    ASSERT(std::cout, k >=0);
	if(n==k or k==0)
		return 0;
	else
        return lfactorial(n)-lfactorial(k)-lfactorial(n-k);
}


static double lcrp(const std::vector<size_t> &Nk, size_t n, double alpha)
{
    ASSERT_GREATER_THAN_ZERO(std::cout, alpha);
    ASSERT_GREATER_THAN_ZERO(std::cout, n);

    double K = double(Nk.size());
    double sum_gammaln = 0;

    for(auto k : Nk)
        sum_gammaln += lgamma(double(k));

    return sum_gammaln + K*log(alpha) + lgamma(alpha) - lgamma(double(n)+alpha);
}


// unnormalized log( P(alpha|k,n)) for crp
static double lcrpUNormPost(size_t k, size_t n, double alpha)
{
    // sometimes alpha is zero because of floating point error
    // ASSERT_GREATER_THAN_ZERO(std::cout, alpha);
    ASSERT_GREATER_THAN_ZERO(std::cout, n);
    ASSERT_GREATER_THAN_ZERO(std::cout, k);

    return lgamma(alpha)+double(k)*log(alpha)-lgamma(alpha+double(n));
}


static double logsumexp(std::vector<double> P)
{
    // if there is a single element in the vector, return that element
    // otherwise there will be log domain problems
    if(P.size() == 1)
        return P.front();

    double max = *std::max_element(P.begin(), P.end());
    double ret = 0;
    for(size_t i = 0; i < P.size(); ++i)
        ret += exp(P[i]-max);

    double retval = log(ret)+max;
    ASSERT_IS_A_NUMBER(std::cout, retval);
    return retval;
}


template<typename lambda>
static long double __simpsons_rule(const lambda &f, double a, double b)
{
    // return (fabs(b-a)/6) * ( f(a) + 4*f((a+b)/2) + f(b) ); // 1/3 rule
    return (fabs(b-a)/8) * (f(a) + 3*f((2*a+b)/3) + 3*f((a+2*b)/3) + f(b)); // 3/8 rule
}


template<typename lambda>
static double __quadrature_recursion(const lambda &f, double a, double b, double eps, double W,
                                     size_t &RECURSION_COUNT, size_t MAX_RECURSIONS)
{
    if(W==0) return 0;

    double c = (a+b)/2;
    double L = __simpsons_rule(f, a, c);
    double R = __simpsons_rule(f, c, b);
    // double err = fabs( 1-(L+R)/W );
    double err = fabs( (L+R)-W );

    if( err <= eps) return L+R;

    ++RECURSION_COUNT;

    // TODO: Return value after max recursions or throw exception?
    if(RECURSION_COUNT > MAX_RECURSIONS)
        throw MaxIterationsReached(RECURSION_COUNT);

    return __quadrature_recursion(f, a, c, eps/2, L, RECURSION_COUNT,  MAX_RECURSIONS) +
           __quadrature_recursion(f, c, b, eps/2, R, RECURSION_COUNT,  MAX_RECURSIONS);
}


template<typename lambda>
static double quadrature(const lambda &f, double a, double b, double eps=0)
{
    if(eps==0){
        // create an error estimate
        double width = fabs(b-a);
        std::vector<double> probe = {f(a+width/3), f(a+width/2), f(a+width*2/3)};
        double max_probe = *std::max_element(probe.begin(),probe.end());
        eps = max_probe*10e-8;
    }

    size_t MAX_RECURSIONS = 1000;  // Maybe too high?
    size_t RECURSION_COUNT = 0;

    double ret = __quadrature_recursion(f, a, b, eps, __simpsons_rule(f, a, b), RECURSION_COUNT,
                                        MAX_RECURSIONS);
    return ret;
}


template<typename lambda_a, typename lambda_b, typename lambda_c>
static double kldivergence(const lambda_a &p, const lambda_b &log_p, lambda_c &log_q,  double a,
                           double b, double eps=0)
{
    auto kl_integral = [&p, &log_p, &log_q](double x){return p(x)*(log_p(x)-log_q(x));};
    return quadrature( kl_integral, a, b, eps);
}

// Calculates discrete KL-divergence given probability distributions p and q. p and q contain log
// probabilities over the same varaibles in the same order.
static double discretekl(std::vector<double> p, std::vector<double> q)
{
    ASSERT_EQUAL(std::cout, p.size(), q.size());

    double kl = 0;
    for(auto &logp_p : p)
        for(auto &logp_q : q)
            kl += exp(logp_p)*(logp_p-logp_q);

    return kl;
}

}} // end namespaces

#endif
