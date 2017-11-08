
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
