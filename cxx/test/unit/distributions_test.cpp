
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
#include <iostream>
#include <vector>

#include "distributions/beta.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/students_t.hpp"
#include "distributions/categorical.hpp"
#include "distributions/symmetric_dirichlet.hpp"


BOOST_AUTO_TEST_SUITE (distributions_test)

// gaussian
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(gaussian_should_add_suffstats){
    double sum_x = 0;
    double sum_x_sq = 0;

    baxcat::dist::gaussian::suffstatInsert(2, sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, 2);
    BOOST_CHECK_EQUAL(sum_x_sq, 4);

    baxcat::dist::gaussian::suffstatInsert(1.1, sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, 2+1.1);
    BOOST_CHECK_EQUAL(sum_x_sq, 4+1.1*1.1);
}

BOOST_AUTO_TEST_CASE(gaussian_should_remove_suffstats){
    double sum_x = 0;
    double sum_x_sq = 0;

    std::vector<double> values = {1,2,3,4};

    baxcat::dist::gaussian::suffstatInsert(values[0], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatInsert(values[1], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatInsert(values[2], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatInsert(values[3], sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, 10);
    BOOST_CHECK_EQUAL(sum_x_sq, 30);

    baxcat::dist::gaussian::suffstatRemove(values[0], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatRemove(values[1], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatRemove(values[2], sum_x, sum_x_sq);
    baxcat::dist::gaussian::suffstatRemove(values[3], sum_x, sum_x_sq);

    BOOST_CHECK_EQUAL(sum_x, 0);
    BOOST_CHECK_EQUAL(sum_x_sq, 0);

}

BOOST_AUTO_TEST_CASE(gaussian_log_pdf_with_suffstats_value_check){
    double log_pdf;

    double n = 10;
    double sum_x = 6.24282197090297;
    double sum_x_sq = 32.0897118215752;

    log_pdf = baxcat::dist::gaussian::logPdfSuffstats(n, sum_x, sum_x_sq, 0.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-25.2342412428343, TOL);

    log_pdf = baxcat::dist::gaussian::logPdfSuffstats(n, sum_x, sum_x_sq, 1.1, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-24.417137074841, TOL);

    log_pdf = baxcat::dist::gaussian::logPdfSuffstats(n, sum_x, sum_x_sq, 0.0, .25);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-20.1320711153431, TOL);
}

BOOST_AUTO_TEST_CASE(gaussian_log_pdf_with_vector_value_check){
    double log_pdf;

    std::vector<double> X = {
        0.537667139546100, 1.833885014595086, -2.258846861003648, 0.862173320368121,
        0.318765239858981, -1.307688296305273, -0.433592022305684, 0.342624466538650,
        3.578396939725760, 2.769437029884877};

    log_pdf = baxcat::dist::gaussian::logPdf(X, 0.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-25.2342412428343, TOL);

    log_pdf = baxcat::dist::gaussian::logPdf(X, 1.1, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-24.417137074841, TOL);

    log_pdf = baxcat::dist::gaussian::logPdf(X, 0.0, .25);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-20.1320711153431, TOL);
}

BOOST_AUTO_TEST_CASE(gaussian_log_pdf_with_single_value_value_check){
    double log_pdf;

    log_pdf = baxcat::dist::gaussian::logPdf(0.0, 0.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-0.918938533204673, TOL);

    log_pdf = baxcat::dist::gaussian::logPdf(0.0, 1.1, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.52393853320467, TOL);

    log_pdf = baxcat::dist::gaussian::logPdf(0.0, 1.1, .25);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.76333571376462, TOL);

    log_pdf = baxcat::dist::gaussian::logPdf(-2.3, 1.1, .25);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-3.05708571376462, TOL);
}

BOOST_AUTO_TEST_CASE(gaussian_cdf_value_check){
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gaussian::cdf(0,0,1), 0.5, TOL );
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gaussian::cdf(2.2,2.2,3.5), 0.5, TOL);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gaussian::cdf(-.2,0,1), 0.420740290560897, TOL );
}

// gamma
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(gamma_log_pdf_with_single_value_value_check){
    double log_pdf;

    log_pdf = baxcat::dist::gamma::logPdf(1.0, 1.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.0, TOL);

    log_pdf = baxcat::dist::gamma::logPdf(1.0, 2.0, 2.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.88629436111989, TOL);

    log_pdf = baxcat::dist::gamma::logPdf(10.2, 2.3, 2.3);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-3.48555901002873, TOL);

    log_pdf = baxcat::dist::gamma::logPdf(10.2, 5, 2.3);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-2.487831172558223, TOL);
}

BOOST_AUTO_TEST_CASE(gamma_log_pdf_with_vector_value_check){
    std::vector<double> X = {
        7.482941841675066, 0.616496511948365, 0.512925679947363,
        2.772692931993200, 2.622390048278854};

    double log_pdf;

    log_pdf = baxcat::dist::gamma::logPdf(X, 1.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-14.0074470138428, TOL);

    log_pdf = baxcat::dist::gamma::logPdf(X, 2.0, 2.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-11.089991075774, TOL);

    log_pdf = baxcat::dist::gamma::logPdf(X, 5, 2);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-28.8418552256724, TOL);
}

BOOST_AUTO_TEST_CASE(gamma_cdf_value_check){
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gamma::cdf(1,1,1), 0.632120558828558, TOL );
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gamma::cdf(2,2,3.5), 0.112585808273521, TOL);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::gamma::cdf(.2,2.3,4.5), 0.000280450981357375, TOL );
}

// categorical
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(categorical_insert_suffstats){
    std::vector<unsigned int> counts(5,0);
    std::vector<unsigned int> values = {0,1,2,3,4,4};

    unsigned int val = 0;

    baxcat::dist::categorical::suffstatInsert(val,counts);
    BOOST_CHECK_EQUAL( counts[0], 1);

    for( auto x : values)
        baxcat::dist::categorical::suffstatInsert(x,counts);

    BOOST_CHECK_EQUAL( counts[0], 2);
    BOOST_CHECK_EQUAL( counts[1], 1);
    BOOST_CHECK_EQUAL( counts[2], 1);
    BOOST_CHECK_EQUAL( counts[3], 1);
    BOOST_CHECK_EQUAL( counts[4], 2);

}

BOOST_AUTO_TEST_CASE(categorical_remove_suffstats){
    std::vector<unsigned int> counts(5,0);
    std::vector<unsigned int> values = {0,0,1,2,3,4,4};


    for( auto x : values)
        baxcat::dist::categorical::suffstatInsert(x,counts);

    BOOST_CHECK_EQUAL( counts[0], 2);
    BOOST_CHECK_EQUAL( counts[1], 1);
    BOOST_CHECK_EQUAL( counts[2], 1);
    BOOST_CHECK_EQUAL( counts[3], 1);
    BOOST_CHECK_EQUAL( counts[4], 2);

    for( auto x : values)
        baxcat::dist::categorical::suffstatRemove(x,counts);

    BOOST_CHECK_EQUAL( counts[0], 0);
    BOOST_CHECK_EQUAL( counts[1], 0);
    BOOST_CHECK_EQUAL( counts[2], 0);
    BOOST_CHECK_EQUAL( counts[3], 0);
    BOOST_CHECK_EQUAL( counts[4], 0 );

}

BOOST_AUTO_TEST_CASE(categorical_log_pdf_single_value_check){

    std::vector<double> p = {.2, .5, .1, .15, .05};

    double pdf;

    pdf = baxcat::dist::categorical::logPdf(1, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, -0.693147180559945, TOL);

    pdf = baxcat::dist::categorical::logPdf(4, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, -2.995732273553991, TOL);
}

BOOST_AUTO_TEST_CASE(categorical_log_pdf_suffstat_value_check){

    std::vector<double> p = {.2,.5,.1,.15,.05};
    std::vector<char> counts = {1,1,1,1,1};

    double pdf, true_val;

    true_val = -9.4980224444279635;
    pdf = baxcat::dist::categorical::logPdfSuffstats(5, counts, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, true_val, TOL);

    counts = {2,2,2,2,2};

    true_val = -18.996044888855927;
    pdf = baxcat::dist::categorical::logPdfSuffstats(10, counts, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, true_val, TOL);
}

BOOST_AUTO_TEST_CASE(categorical_log_pdf_vector_value_check){

    std::vector<double> p = {.2,.5,.1,.15,.05};
    std::vector<char> X = {0,1,2,3,4};

    double pdf, true_val;

    true_val = -9.4980224444279635;
    pdf = baxcat::dist::categorical::logPdf(X, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, true_val, TOL);

    X = {0,1,2,3,4,0,1,2,3,4};

    true_val = -18.996044888855927;
    pdf = baxcat::dist::categorical::logPdf(X, p);
    BOOST_CHECK_CLOSE_FRACTION( pdf, true_val, TOL);

}

// beta
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(beta_should_add_suffstats){
    double sum_log_x = 0;
    double sum_log_minus_x = 0;

    baxcat::dist::beta::suffstatInsert(.25, sum_log_x, sum_log_minus_x);

    BOOST_CHECK_CLOSE_FRACTION(sum_log_x, log(.25), TOL);
    BOOST_CHECK_CLOSE_FRACTION(sum_log_minus_x, log(1-.25), TOL);

    baxcat::dist::beta::suffstatInsert(.1, sum_log_x, sum_log_minus_x);

    BOOST_CHECK_CLOSE_FRACTION(sum_log_x, log(.25)+log(.1), TOL);
    BOOST_CHECK_CLOSE_FRACTION(sum_log_minus_x, log(1-.25)+log(1-.1), TOL);
}

BOOST_AUTO_TEST_CASE(beta_should_remove_suffstats){
    double sum_log_x = 0;
    double sum_log_minus_x = 0;

    std::vector<double> values = {.1,.2,.5,.75};

    baxcat::dist::beta::suffstatInsert(values[0], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatInsert(values[1], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatInsert(values[2], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatInsert(values[3], sum_log_x, sum_log_minus_x);

    BOOST_CHECK_CLOSE_FRACTION(sum_log_x, -4.89285225843987, TOL);
    BOOST_CHECK_CLOSE_FRACTION(sum_log_minus_x, -2.40794560865187, TOL);

    baxcat::dist::beta::suffstatRemove(values[0], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatRemove(values[1], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatRemove(values[2], sum_log_x, sum_log_minus_x);
    baxcat::dist::beta::suffstatRemove(values[3], sum_log_x, sum_log_minus_x);

    BOOST_CHECK_SMALL(sum_log_x, TOL);
    BOOST_CHECK_SMALL(sum_log_minus_x, TOL);
}

BOOST_AUTO_TEST_CASE(beta_log_pdf_single_value_checks){

    double log_pdf;
    log_pdf = baxcat::dist::beta::logPdf(.1, 1.0, 1.0);
    BOOST_CHECK_SMALL(log_pdf, TOL);

    log_pdf = baxcat::dist::beta::logPdf(.1, .5, .5);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, 0.059242918476536, TOL);

    log_pdf = baxcat::dist::beta::logPdf(.5, .5, .5);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -0.451582705289455, TOL);

    log_pdf = baxcat::dist::beta::logPdf(.5, 3.5, .1);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -3.24979843440078, TOL);
}

BOOST_AUTO_TEST_CASE(beta_log_pdf_suffstats_value_checks){
    double sum_log_x = -4.89285225843987;
    double sum_log_minus_x = -2.40794560865187;

    double log_pdf;
    log_pdf = baxcat::dist::beta::logPdfSuffstats(4.0, sum_log_x, sum_log_minus_x, 1.0, 1.0);
    BOOST_CHECK_SMALL(log_pdf, TOL);

    log_pdf = baxcat::dist::beta::logPdfSuffstats(4.0, sum_log_x, sum_log_minus_x, .5, .5);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -0.928520609851728, TOL);

    log_pdf = baxcat::dist::beta::logPdfSuffstats(4.0, sum_log_x, sum_log_minus_x, 3.5, .1);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -18.6280313803324, TOL);
}

BOOST_AUTO_TEST_CASE(beta_log_pdf_vector_value_checks){

    std::vector<double> X = {.1,.2,.5,.75};
    double log_pdf;

    log_pdf = baxcat::dist::beta::logPdf(X, 1.0, 1.0);
    BOOST_CHECK_SMALL(log_pdf, TOL);

    log_pdf = baxcat::dist::beta::logPdf(X, .5, .5);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -0.928520609851728, TOL);

    log_pdf = baxcat::dist::beta::logPdf(X, 3.5, .1);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf, -18.6280313803324, TOL);
}

BOOST_AUTO_TEST_CASE(beta_cdf_value_check){
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::beta::cdf(.5,1,1), 0.5, TOL );
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::beta::cdf(.5,.5,.5), 0.5, TOL);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::beta::cdf(.5,5,5), 0.5, TOL );
    BOOST_CHECK_CLOSE_FRACTION( baxcat::dist::beta::cdf(.5,5,.5), 0.0101195597354337, TOL );
}


// student's t
// `````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(t_log_pdf_with_single_value_value_check){
    double log_pdf;

    log_pdf = baxcat::dist::students_t::logPdf(1.0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.83787706640935, TOL);

    log_pdf = baxcat::dist::students_t::logPdf(0, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-1.1447298858494, TOL);

    log_pdf = baxcat::dist::students_t::logPdf(2.4,1.85);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-3.0642285740553, TOL);

}

BOOST_AUTO_TEST_CASE(t_log_pdf_with_vector_value_check){
    std::vector<double> X = {
        -0.909948773540185, 1.025918923784846, 1.761090386120126,
        0.669939425343656, -1.873966046190918, 0.145845574764761};

    double log_pdf;

    log_pdf = baxcat::dist::students_t::logPdf(X, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-11.5004880388291, TOL);

    log_pdf = baxcat::dist::students_t::logPdf(X, 2.3);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-10.5250620734235, TOL);

    log_pdf = baxcat::dist::students_t::logPdf(X, .2);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,-16.0173232819958, TOL);
}

// Dirichlet
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(symmetric_dirichlet_value_check){
    std::vector<double> p = {.2, .3, .5};

    double log_pdf;

    log_pdf = baxcat::dist::symmetric_dirichlet::logPdf(p, 1.0);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,0.693147180559945, TOL);

    log_pdf = baxcat::dist::symmetric_dirichlet::logPdf(p, 2.3);
    BOOST_CHECK_CLOSE_FRACTION(log_pdf,1.37165082501073, TOL);
}

BOOST_AUTO_TEST_SUITE_END()
