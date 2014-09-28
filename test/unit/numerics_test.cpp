
// BaxCat: an extensible cross-catigorization engine.
// Copyright (C) 2014 Baxter Eaves
//
// This program is free software: you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License (LICENSE.txt) along with this
// program. If not, see <http://www.gnu.org/licenses/>.
//
// You may contact the mantainers of this software via github
// <https://github.com/BaxterEaves/baxcat_cxx>.

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/math/distributions/beta.hpp>
#include <boost/math/distributions/normal.hpp>
#include <iostream>
#include <vector>
#include <cmath>

#include "distributions/gaussian.hpp"
#include "numerics.hpp"

BOOST_AUTO_TEST_SUITE (numerics_test)

// log factorial
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_factorial_value_checks){
    // fringe case 0
    double ans;
    ans = baxcat::numerics::lfactorial(0);
    BOOST_CHECK_CLOSE_FRACTION(ans,0,TOL);

    // fringe case 1
    ans = baxcat::numerics::lfactorial(1);
    BOOST_CHECK_CLOSE_FRACTION(ans,0,TOL);

    ans = baxcat::numerics::lfactorial(10);
    BOOST_CHECK_CLOSE_FRACTION(ans,15.1044125730755,TOL);
}

// log nchoosek
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_nchoosek_value_checks){
    // fringe case 0
    double ans;
    // fringe case 0
    ans = baxcat::numerics::lnchoosek(10,0);
    BOOST_CHECK_CLOSE_FRACTION(ans,0,TOL);

    // fringe case n==k
    ans = baxcat::numerics::lnchoosek(10,10);
    BOOST_CHECK_CLOSE_FRACTION(ans,0,TOL);

    // case k = 1
    ans = baxcat::numerics::lnchoosek(10,1);
    BOOST_CHECK_CLOSE_FRACTION(ans,2.30258509299405,TOL);

    ans = baxcat::numerics::lnchoosek(8,5);
    BOOST_CHECK_CLOSE_FRACTION(ans,4.02535169073515,TOL);
}

// log beta function
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_beta_value_checks){
    // fringe case 0
    double ans;
    // fringe case a=b=0
    ans = baxcat::numerics::lbeta(1,1);
    BOOST_CHECK_CLOSE_FRACTION(ans,0,TOL);

    ans = baxcat::numerics::lbeta(2,2);
    BOOST_CHECK_CLOSE_FRACTION(ans,-1.79175946922805,TOL);

    ans = baxcat::numerics::lbeta(20.5,1.2);
    BOOST_CHECK_CLOSE_FRACTION(ans,-3.71567154635848,TOL);

    ans = baxcat::numerics::lbeta(1.2,20.5);
    BOOST_CHECK_CLOSE_FRACTION(ans,-3.71567154635848,TOL);
}

// quadrature (adaptive Simpson's rule)
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(quadrature_value_checks_against_beta_distribution){
    const double QUADTOL = 10E-5;

    double quad_epsilon = 10E-10;

    auto betajeff = [](double x){
        boost::math::beta_distribution<double> beta(.5,.5);
        return pdf(beta,x);
    };

    double cdf = baxcat::numerics::quadrature(betajeff,10E-10,.5, quad_epsilon);
    BOOST_CHECK_CLOSE_FRACTION(cdf,.5, QUADTOL);

    cdf = baxcat::numerics::quadrature(betajeff,10E-10,1-10E-10, quad_epsilon);
    BOOST_CHECK_CLOSE_FRACTION(cdf,1,QUADTOL);

    // mixture of normal distributions
    auto normix = [](double x){
        boost::math::normal_distribution<double> norm_1(-3,1);
        boost::math::normal_distribution<double> norm_2(3,1);
        return .5*pdf(norm_1,x) + .5*pdf(norm_2,x);
    };

    cdf = baxcat::numerics::quadrature(normix,-50,0, quad_epsilon);
    BOOST_CHECK_CLOSE_FRACTION(cdf,.5,QUADTOL);

    cdf = baxcat::numerics::quadrature(normix,-25,25, quad_epsilon);
    BOOST_CHECK_CLOSE_FRACTION(cdf,1,QUADTOL);

    cdf = baxcat::numerics::quadrature(normix,-4,-2, quad_epsilon);
    BOOST_CHECK_CLOSE_FRACTION(cdf,0.341344889186363,QUADTOL);
}

// KL divergence
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(test_kl_divergence_values_against_normal_closed_form){
    double mean_p = 0;
    double mean_q = .5;

    auto exp_p = [mean_p](double x){
        return exp(baxcat::dist::gaussian::logPdf(x, mean_p, 1));
    };
    auto log_p = [mean_p](double x){
        return baxcat::dist::gaussian::logPdf(x, mean_p, 1);
    };
    auto log_q = [mean_q](double x){
        return baxcat::dist::gaussian::logPdf(x, mean_q, 1);
    };

    double estimated_kl = baxcat::numerics::kldivergence(exp_p, log_p, log_q,  -10, 10);
    double true_kl = .5*(mean_p-mean_q)*(mean_p-mean_q); // simplified because stddevs are 1

    BOOST_CHECK_CLOSE_FRACTION(estimated_kl, true_kl, TOL);
}

// lcrp
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(lcrp_value_checks){
    std::vector<size_t> Nk = {1,1,1,1};
    unsigned int N = 4;
    double alpha = 1.0;

    double ans = baxcat::numerics::lcrp(Nk, N, alpha);
    BOOST_CHECK_CLOSE_FRACTION( ans, -3.17805383034795, TOL);

    alpha = 2.1;
    ans = baxcat::numerics::lcrp(Nk, N, alpha);
    BOOST_CHECK_CLOSE_FRACTION( ans, -1.94581759074351, TOL);

}

// logsumexp
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(logsumexp_single_value_should_be_unchanged){
    std::vector<double> v = {-1.0};
    double val = baxcat::numerics::logsumexp(v);
    BOOST_CHECK_CLOSE_FRACTION( val, v.front(), TOL);
}

BOOST_AUTO_TEST_CASE(logsumexp_value_checks){
    std::vector<double> v1 = {-1.0, -2.0, -3.0};
    double val = baxcat::numerics::logsumexp(v1);
    BOOST_CHECK_CLOSE_FRACTION(val, -0.59239403555562, TOL);

    std::vector<double> v2 = {-350, -210, -220};
    val = baxcat::numerics::logsumexp(v2);
    BOOST_CHECK_CLOSE_FRACTION(val, -209.999954601101, TOL);

    std::vector<double> v3 = {-11.64,-11.363,-11.136,-10.976,-10.894,-10.896,
                              -10.982,-11.146,-11.375,-11.655,-11.971};
    val = baxcat::numerics::logsumexp(v3);
    BOOST_CHECK_CLOSE_FRACTION(val, -8.82420447065017, TOL);
}


BOOST_AUTO_TEST_SUITE_END()
