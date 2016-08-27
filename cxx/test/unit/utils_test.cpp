
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <vector>

#include "utils.hpp"

#define TOL 10e-8

BOOST_AUTO_TEST_SUITE (utils_test)

using namespace std;


// test binary_search
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(binary_search_should_find_existing_elements_cont)
{
    std::vector<size_t> keys = {0,1,2,3,4};
    size_t idx;
    for( size_t key: keys){
        idx = baxcat::utils::binary_search(keys, key);
        BOOST_CHECK_EQUAL( idx, key);
    }
}


BOOST_AUTO_TEST_CASE(binary_search_should_find_existing_elements_discont)
{
    std::vector<size_t> keys = {1,3,7,14,21};
    size_t idx;
    idx = baxcat::utils::binary_search(keys, 1);
    BOOST_CHECK_EQUAL( idx, 0);
    idx = baxcat::utils::binary_search(keys, 3);
    BOOST_CHECK_EQUAL( idx, 1);
    idx = baxcat::utils::binary_search(keys, 7);
    BOOST_CHECK_EQUAL( idx, 2);
    idx = baxcat::utils::binary_search(keys, 14);
    BOOST_CHECK_EQUAL( idx, 3);
    idx = baxcat::utils::binary_search(keys, 21);
    BOOST_CHECK_EQUAL( idx, 4);
}


BOOST_AUTO_TEST_CASE(binary_search_should_tell_where_to_insert_missing_vals)
{
    std::vector<size_t> keys = {1,3,7,14,21};
    size_t idx;
    idx = baxcat::utils::binary_search(keys, 2);
    BOOST_CHECK_EQUAL( idx, 1);

    idx = baxcat::utils::binary_search(keys, 4);
    BOOST_CHECK_EQUAL( idx, 2);
    idx = baxcat::utils::binary_search(keys, 6);
    BOOST_CHECK_EQUAL( idx, 2);

    idx = baxcat::utils::binary_search(keys, 8);
    BOOST_CHECK_EQUAL( idx, 3);
    idx = baxcat::utils::binary_search(keys, 13);
    BOOST_CHECK_EQUAL( idx, 3);

    idx = baxcat::utils::binary_search(keys, 15);
    BOOST_CHECK_EQUAL( idx, 4);
    idx = baxcat::utils::binary_search(keys, 20);
    BOOST_CHECK_EQUAL( idx, 4);
}


BOOST_AUTO_TEST_CASE(binary_search_should_tell_where_to_insert_edge_vals)
{
    std::vector<size_t> keys = {1,3,7,14,21};
    size_t idx;
    idx = baxcat::utils::binary_search(keys, 0);
    BOOST_CHECK_EQUAL( idx, 0);
    idx = baxcat::utils::binary_search(keys, 22);
    BOOST_CHECK_EQUAL( idx, 5);
    idx = baxcat::utils::binary_search(keys, 100);
    BOOST_CHECK_EQUAL( idx, 5);
}


// test linspace
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(linspace_vec_should_be_length_n)
{

    size_t n = 10;
    std::vector<double> vec = baxcat::utils::linspace(1,2,n);
    BOOST_CHECK( vec.size() == n );

    // edge case (minimum number of entries)
    n = 2;
    vec = baxcat::utils::linspace(1,2,n);
    BOOST_CHECK( vec.size() == n );
}


BOOST_AUTO_TEST_CASE(linspace_first_and_last_entry_should_match_a_and_b)
{

    double a = 0;
    double b = 10;

    std::vector<double> vec = baxcat::utils::linspace(a,b,10);
    BOOST_CHECK(vec.front() == a);
    BOOST_CHECK(vec.back() == b);

    // edge case (minum number of entries);
    vec = baxcat::utils::linspace(a,b,2);
    BOOST_CHECK(vec.front() == a);
    BOOST_CHECK(vec.back() == b);

}


BOOST_AUTO_TEST_CASE(linspace_entries_should_strictly_increase)
{
    double a = 1;
    double b = 20;

    std::vector<double> vec = baxcat::utils::linspace(a,b,100);

    for(int i = 1; i < 100; i++)
        BOOST_CHECK( vec[i-1] < vec[i] );
}


// test log_linspace
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_linspace_vec_should_be_length_n)
{

    unsigned int n = 10;
    std::vector<double> vec = baxcat::utils::log_linspace(1,2,n);
    BOOST_CHECK( vec.size() == n );

    // edge case (minimum number of entries)
    n = 2;
    vec = baxcat::utils::log_linspace(1,2,n);
    BOOST_CHECK( vec.size() == n );
}


BOOST_AUTO_TEST_CASE(log_linspace_first_and_last_entry_should_match_a_and_b)
{

    double a = 1;
    double b = 10;

    std::vector<double> vec = baxcat::utils::log_linspace(a,b,10);
    BOOST_CHECK_CLOSE_FRACTION(vec.front(),  a, TOL);
    BOOST_CHECK_CLOSE_FRACTION(vec.back(),  b, TOL);

    // edge case (minum number of entries);
    vec = baxcat::utils::log_linspace(a,b,2);
    BOOST_CHECK_CLOSE_FRACTION(vec.front(),  a, TOL);
    BOOST_CHECK_CLOSE_FRACTION(vec.back(),  b, TOL);
}


BOOST_AUTO_TEST_CASE(log_linspace_entries_should_strictly_increase)
{
    double a = 1;
    double b = 20;

    std::vector<double> vec = baxcat::utils::log_linspace(a,b,100);

    for(int i = 1; i < 100; i++)
        BOOST_CHECK( vec[i-1] < vec[i] );
}


BOOST_AUTO_TEST_CASE(log_linspace_intervals_should_strictly_increase)
{
    double a = 1;
    double b = 20;

    std::vector<double> vec = baxcat::utils::log_linspace(a,b,100);

    for(int i = 2; i < 100; i++){
        double interval_0 = vec[i-1]-vec[i-2];
        double interval_1 = vec[i]-vec[i-1];
        BOOST_CHECK(interval_0 < interval_1 );
    }
}


BOOST_AUTO_TEST_CASE(log_linspace_a_is_0_should_be_ok)
{
    double a = 0;

    std::vector<double> vec = baxcat::utils::log_linspace(a,1,10);
    BOOST_CHECK_CLOSE_FRACTION(a, vec.front(), TOL);
}


// test min and max
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(min_should_return_min_element)
{
    std::vector<double> v = {-1,3,5,2,2};
    double x = baxcat::utils::vector_min(v);
    BOOST_CHECK(x == -1);

    v = {3,5,2,2,-2};
    x = baxcat::utils::vector_min(v);
    BOOST_CHECK(x == -2);

   v = {3,5,2,-3,1,4};
   x = baxcat::utils::vector_min(v);
   BOOST_CHECK(x == -3);
}


BOOST_AUTO_TEST_CASE(max_should_return_max_element)
{
    std::vector<double> v = {10,3,5,2,2};
    double x = baxcat::utils::vector_max(v);
    BOOST_CHECK(x == 10);

    v = {3,5,2,2,20};
    x = baxcat::utils::vector_max(v);
    BOOST_CHECK(x == 20);

   v = {3,5,2,30,1,4};
   x = baxcat::utils::vector_max(v);
   BOOST_CHECK(x == 30);
}


BOOST_AUTO_TEST_CASE(argmax_should_return_max_index)
{
    std::vector<double> v = {10,3,5,2,2};
    auto x = baxcat::utils::argmax(v);
    BOOST_CHECK(x == 0);

    v = {3,5,2,2,20};
    x = baxcat::utils::argmax(v);
    BOOST_CHECK(x == 4);

   v = {3,5,2,30,1,4};
   x = baxcat::utils::argmax(v);
   BOOST_CHECK(x == 3);
}


// test vector mean
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(mean_value_checks)
{
    std::vector<double> a = {0,0,0,0,0};
    double ma = baxcat::utils::vector_mean(a);
    BOOST_CHECK_CLOSE_FRACTION(ma, 0.0, 10E-8);

    std::vector<double> b = {1,2,3,4,5};
    double mb = baxcat::utils::vector_mean(b);
    BOOST_CHECK_CLOSE_FRACTION(mb, 3.0, 10E-8);

    std::vector<double> c = {1};
    double mc = baxcat::utils::vector_mean(c);
    BOOST_CHECK_CLOSE_FRACTION(mc, 1, 10E-8);
}


// test sum of squares
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(sum_of_squares_value_checks)
{
    std::vector<double> a = {0,0,0,0,0};
    double va = baxcat::utils::sum_of_squares(a);
    BOOST_CHECK_CLOSE_FRACTION(va, 0.0, 10E-8);

    std::vector<double> b = {1,2,3,4,5};
    double vb = baxcat::utils::sum_of_squares(b);
    BOOST_CHECK_CLOSE_FRACTION(vb, 10.0, 10E-8);
}


// test sum
//`````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(sum_should_work_four_doubles)
{
    std::vector<double> a = {0,0,0,0,0};
    double va = baxcat::utils::sum(a);
    BOOST_CHECK_CLOSE_FRACTION(va, 0.0, 10E-8);

    std::vector<double> b = {1,2,3,4,5};
    double vb = baxcat::utils::sum(b);
    BOOST_CHECK_CLOSE_FRACTION(vb, 15.0, 10E-8);
}


BOOST_AUTO_TEST_CASE(sum_should_work_four_uint)
{
    std::vector<size_t> a = {0,0,0,0,0};
    size_t va = baxcat::utils::sum(a);
    BOOST_CHECK_EQUAL(va, 0);

    std::vector<size_t> b = {1,2,3,4,5};
    size_t vb = baxcat::utils::sum(b);
    BOOST_CHECK_EQUAL(vb, 15);
}
BOOST_AUTO_TEST_SUITE_END()
