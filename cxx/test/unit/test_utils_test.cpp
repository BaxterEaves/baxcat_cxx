
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
#include <vector>
#include <iostream>
#include "distributions/gaussian.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE (test_utils_test)

using std::vector;

// has_element
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(has_element_should_return_1_if_true){
    std::vector<int> a = {0,1,2,3,4};
    BOOST_CHECK( baxcat::test_utils::hasElement(a, 0) == 1 );
    BOOST_CHECK( baxcat::test_utils::hasElement(a, 3) == 1 );
    BOOST_CHECK( baxcat::test_utils::hasElement(a, 4) == 1 );
}
BOOST_AUTO_TEST_CASE(has_element_should_return_neg_1_if_empty){
    std::vector<int> a;
    BOOST_CHECK( baxcat::test_utils::hasElement(a, 10) == -1);
}
BOOST_AUTO_TEST_CASE(has_element_shouls_return_0_if_false){
   std::vector<int> a = {0,1,3,4};
   BOOST_CHECK( baxcat::test_utils::hasElement(a, 2) == 0 );
   BOOST_CHECK( baxcat::test_utils::hasElement(a, 10) == 0 );
}

// are_identical
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(are_identical_should_return_neg_1_if_diff_size){
    std::vector<int> a = {0,1,2,3,4};
    std::vector<int> b = {0,1,2,3};
    BOOST_CHECK( baxcat::test_utils::areIdentical(a,b) == -1);
    BOOST_CHECK( baxcat::test_utils::areIdentical(b,a) == -1);
}
BOOST_AUTO_TEST_CASE(are_identical_should_return_1_if_true){
    std::vector<int> a = {0,1,2,3,4};
    std::vector<int> b = {0,1,2,3,4};
    BOOST_CHECK( baxcat::test_utils::areIdentical(a,b) == 1);
    BOOST_CHECK( baxcat::test_utils::areIdentical(b,a) == 1);
}
BOOST_AUTO_TEST_CASE(are_identical_should_return_0_if_false){
    std::vector<int> a = {0,1,2,3,4};
    std::vector<int> b = {1,0,2,3,4};
    BOOST_CHECK( baxcat::test_utils::areIdentical(a,b) == 0);
    BOOST_CHECK( baxcat::test_utils::areIdentical(b,a) == 0);
}

// has_same_elements
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(has_same_elements_should_return_0_if_false){
    std::vector<int> a = {0,1,2,3,4};
    std::vector<int> b = {0,0,2,3,4};
    BOOST_CHECK( baxcat::test_utils::hasSameElements(a,b) == 0);
    BOOST_CHECK( baxcat::test_utils::hasSameElements(b,a) == 0);
    std::vector<int> c = {0,1,2,3,3};
    BOOST_CHECK( baxcat::test_utils::hasSameElements(a,c) == 0);
    BOOST_CHECK( baxcat::test_utils::hasSameElements(c,a) == 0);
}
BOOST_AUTO_TEST_CASE(has_same_elements_should_return_1_if_true){
    std::vector<int> a = {0,1,2,3,4};
    std::vector<int> b = {0,1,2,3,4};
    BOOST_CHECK( baxcat::test_utils::hasSameElements(a,b) == 1);
    BOOST_CHECK( baxcat::test_utils::hasSameElements(b,a) == 1);
    std::vector<int> c = {4,2,1,3,0};
    BOOST_CHECK( baxcat::test_utils::hasSameElements(a,c) == 1);
    BOOST_CHECK( baxcat::test_utils::hasSameElements(c,a) == 1);
}

// 2 sample ks-test
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(ks_test2_value_check_1){
    std::vector<double> x1 = {
        0.252124808964812, -0.771478905272172, 0.919545458360691, 0.852157548449778,
        0.657126189710851, -0.753464340423204, -1.338501825721383, 0.652225083883824,
        1.447216675248885, -1.290125974033266, -2.208189200045856, 1.436108116509668};

    std::vector<double> x2 = {
        -0.061672202755906, 1.177806454354101, 0.985513417813457, -1.218578395542881,
        -0.431878290069235, -0.834941294297264, 0.179680103648401, 1.166302425034432,
        0.055985942286178, -2.100041845720212, 1.235297089420208, 0.002671527009520};

    BOOST_CHECK_CLOSE_FRACTION (baxcat::test_utils::twoSampleKSTest(x1,x2), .25, TOL);
}

BOOST_AUTO_TEST_CASE(ks_test2_value_check_2){
    std::vector<double> x1 = {
        -0.459536791820333, -2.103176591870821, 0.373921684965675, 0.245180507935031,
        0.338579197802804, -1.078065162319935, -0.730162332452103, -0.916327269903331,
        1.787553175402401, -0.820403529197700, -0.196711337853968, -0.890143741968409,
        0.910748613628087, -0.012265249810327, 0.072814482378082};

    std::vector<double> x2 = {
        -0.061672202755906, 1.177806454354101, 0.985513417813457, -1.218578395542881,
        -0.431878290069235, -0.834941294297264, 0.179680103648401, 1.166302425034432,
        0.055985942286178, -2.100041845720212, 1.235297089420208, 0.002671527009520};

    BOOST_CHECK_CLOSE_FRACTION (baxcat::test_utils::twoSampleKSTest(x1,x2), 0.266666666666667, TOL);
}

// one sample ks-test
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(ks_test1_value_check_1){
    std::vector<double> X = {
        1.242122872242683, 2.081656873166800, 2.487197859042452, -0.763281987903164,
        -1.934878784026592, 6.117383037519073, -1.108205808796507, -1.420813408262703,
        0.956973901836422, -2.165479220537083, -1.343605692970439, -3.076040476612111};

    auto cdf = [](double x){return baxcat::dist::gaussian::cdf(x,0,1);};

    BOOST_CHECK_CLOSE_FRACTION(baxcat::test_utils::oneSampleKSTest(X,cdf), 0.366113528630429, TOL);
}

BOOST_AUTO_TEST_CASE(ks_test1_value_check_2){
    std::vector<double> X = {
        -0.459536791820333, -2.103176591870821, 0.373921684965675, 0.245180507935031,
        0.338579197802804, -1.078065162319935, -0.730162332452103, -0.916327269903331,
        1.787553175402401, -0.820403529197700, -0.196711337853968, -0.890143741968409,
        0.910748613628087, -0.012265249810327, 0.072814482378082};

    auto cdf = [](double x){return baxcat::dist::gaussian::cdf(x,0,1);};

    BOOST_CHECK_CLOSE_FRACTION(baxcat::test_utils::oneSampleKSTest(X,cdf), 0.2208979579366, TOL);
}

BOOST_AUTO_TEST_CASE(ks_test1_value_check_3){
    std::vector<double> X = {
        -0.039046130259433, -0.875950199996807, 0.436696622718939, -2.960899999365033,
        -1.197698225974150, -2.207845485259799, 1.908008030729362, -0.174781105771509,
        0.378971977916614, -2.058180257987361, -1.468615581100624, -1.272469409250188};

    auto cdf = [](double x){return baxcat::dist::gaussian::cdf(x,0,1);};

    BOOST_CHECK_CLOSE_FRACTION(baxcat::test_utils::oneSampleKSTest(X,cdf), 0.392804779665865, TOL);
}

// ks-test null hypothesis rejection
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(ks_test_reject_null_should_reject_high_ks_stat){
    double ks_stat = 0.72;
    bool reject_null = baxcat::test_utils::ksTestRejectNull(ks_stat, 100,100);
    BOOST_CHECK( reject_null );
}

BOOST_AUTO_TEST_CASE(ks_test_reject_null_should_accept_low_ks_stat){
    double ks_stat = 0.11;
    bool reject_null = baxcat::test_utils::ksTestRejectNull(ks_stat, 100,100);
    BOOST_CHECK( !reject_null );
}

// chi-square test
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(chi2_test_with_observed_and_expected_value_check){
    vector<double> observed = {6, 16, 10, 12, 4, 2};
    vector<double> expected = {
        8.33333333333333, 8.33333333333333, 8.33333333333333,
        8.33333333333333, 8.33333333333333, 8.33333333333333};

    double chi2_stat = baxcat::test_utils::chi2Stat(observed,expected);
    BOOST_CHECK_CLOSE_FRACTION(chi2_stat, 16.72, TOL);

    // against non uniform expected
    expected = {
        14.62340425837566826317, 14.55636808229502676681, 7.34734777520420045960,
        10.96603656437204321605, 1.46493344609166276094, 1.04190987366140119796};

    chi2_stat = baxcat::test_utils::chi2Stat(observed,expected);
    BOOST_CHECK_CLOSE_FRACTION(chi2_stat, 11.551520129390337, TOL);
}

BOOST_AUTO_TEST_CASE(chi2_test_from_data_value_check){
    vector<size_t> X = {1, 3, 1, 2, 0, 3, 1, 2, 2, 3, 3, 2, 0, 0, 1, 3, 1, 3, 0, 3, 1, 0, 1, 2, 1};
    vector<double> observed = {5, 8, 5, 7};
    vector<double> expected = {6.25, 6.25, 6.25, 6.25};

    double chi2_a = baxcat::test_utils::chi2Stat(X);
    double chi2_b = baxcat::test_utils::chi2Stat(observed, expected);
    BOOST_CHECK_CLOSE_FRACTION(chi2_b, 1.08, TOL);
    BOOST_CHECK_CLOSE_FRACTION(chi2_a, chi2_b, TOL);
}

BOOST_AUTO_TEST_SUITE_END()
