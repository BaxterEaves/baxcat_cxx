#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>
#include <cassert>

#include "test_utils.hpp"
#include "utils.hpp"
#include "prng.hpp"
#include "omp.h"

BOOST_AUTO_TEST_SUITE(baxcat_rng_test)

BOOST_AUTO_TEST_CASE(should_generate_same_number_with_same_seed) {

    static baxcat::PRNG prng_1(10);
    std::uniform_int_distribution<unsigned int> dist;

    unsigned int r1 = dist(prng_1.getRNG());

    baxcat::PRNG prng_2(10);
    unsigned int r2 = dist(prng_2.getRNG());

    BOOST_REQUIRE_EQUAL(r1, r2);

}

BOOST_AUTO_TEST_CASE(subsequent_numbers_should_not_be_equal){
    static baxcat::PRNG prng(10);
    std::uniform_int_distribution<unsigned int> dist;

    unsigned int r1, r2;
    r1 = dist(prng.getRNG());
    r2 = dist(prng.getRNG());

    BOOST_CHECK( r1 != r2 );
}

BOOST_AUTO_TEST_CASE(number_from_other_threads_should_not_be_equal){
    static baxcat::PRNG prng(10);
    std::uniform_int_distribution<unsigned int> dist;

    unsigned int r1, r2;
    r1 = dist(prng.getRNGByIndex(0));
    r2 = dist(prng.getRNGByIndex(1));

    BOOST_CHECK( r1 != r2);
}

BOOST_AUTO_TEST_CASE(number_from_edge_thread_should_be_same_with_seed){

    unsigned int last_thread = omp_get_max_threads()-1;
    unsigned int r0a, r4a;
    unsigned int r0b, r4b;

    // can't test a parallel RNG if we aren't in a parallel environment
    BOOST_REQUIRE(last_thread > 0 );

    static baxcat::PRNG prng(10);
    static baxcat::PRNG prng_2(10);

    std::uniform_int_distribution<unsigned int>  dist;

    r0a = dist(prng.getRNGByIndex(0));
    r0b = dist(prng_2.getRNGByIndex(0));

    r4a = dist(prng.getRNGByIndex(last_thread));
    r4b = dist(prng_2.getRNGByIndex(last_thread));

    BOOST_CHECK_EQUAL(r0a, r0b);
    BOOST_CHECK_EQUAL(r4a, r4b);
}

// Test random element
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(random_element_should_return_an_element){
    static baxcat::PRNG rng(10); 
    std::vector<int> a = {1,2,3,4,5};
    auto element = rng.randomElement(a);
    BOOST_CHECK( baxcat::test_utils::hasElement(a, element) );
}

// Test Shuffle
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(shuffle_should_return_right_size_vector_int){
    static baxcat::PRNG rng(10);
    std::vector<int> a = {1,2,3,4,5};
    std::vector<int> b = rng.shuffle(a);
    BOOST_CHECK(a.size() == b.size() );
}
BOOST_AUTO_TEST_CASE(shuffle_should_produce_different_vectors){
    static baxcat::PRNG rng(10);
    std::vector<int> a = {1,2,3,4,5};
    std::vector<int> b = rng.shuffle(a);
    BOOST_CHECK( baxcat::test_utils::areIdentical(a,b) == 0 );
}
BOOST_AUTO_TEST_CASE(shuffled_vector_should_have_same_elements){
    static baxcat::PRNG rng(10);
    std::vector<int> a = {1,2,3,4,5};
    std::vector<int> b = rng.shuffle(a);
    BOOST_CHECK( baxcat::test_utils::hasSameElements(a,b) == 1 );
}

// Test pflip
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(pflip_should_return_valid_index_normalized){
    static baxcat::PRNG rng(10);
    std::vector<double> a = { .25, .25, .25, .25};
    auto index = rng.pflip(a);
    BOOST_CHECK(index >= 0 && index < a.size());
}
BOOST_AUTO_TEST_CASE(pflip_should_return_valid_index_unnormalized){
    static baxcat::PRNG rng(10);
    std::vector<double> a = { 1, 1, 1, 1};
    auto index = rng.pflip(a);
    BOOST_CHECK(index >= 0 && index < a.size());
}
BOOST_AUTO_TEST_CASE(pflip_should_return_deteministic_one_entry){
    static baxcat::PRNG rng(10);
    std::vector<double> a = {1};
    for(int i = 0; i < 100; i++){
        auto index = rng.pflip(a);
        BOOST_CHECK(index == 0 );
    }
}
BOOST_AUTO_TEST_CASE(pflip_should_return_determistic_determinsitic_vector){
    static baxcat::PRNG rng(10);
    std::vector<double> a = {0,0,1,0};
    for(int i = 0; i < 100; i++){
        auto index = rng.pflip(a);
        BOOST_CHECK(index == 2);
    }
}

// Test log pflip
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(log_pflip_should_return_valid_index_unnormalized){
    static baxcat::PRNG rng(10);
    std::vector<double> a = { -12, -12, -12, -12};
    auto index = rng.lpflip(a);
    BOOST_CHECK(index >= 0 && index < a.size());
}
BOOST_AUTO_TEST_CASE(log_pflip_should_return_deteministic_one_entry){
    static baxcat::PRNG rng(10);
    std::vector<double> a = {-12};
    for(int i = 0; i < 100; i++){
        auto index = rng.lpflip(a);
        BOOST_CHECK(index == 0 );
    }
}

// Test crp gen
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(crp_gen_pieces_should_agree){
    std::vector<size_t> Z;
    std::vector<size_t> Nk;
    size_t K;

    static baxcat::PRNG rng(10);

    rng.crpGen(1.0, 100, Z, K, Nk);

    BOOST_CHECK_EQUAL( Z.size(), 100);
    BOOST_CHECK_EQUAL( Nk.size(), K);

    size_t sum_Nk = 0;
    for(size_t k : Nk)
        sum_Nk += k;

    BOOST_CHECK_EQUAL( sum_Nk, 100);
}

// Test the random distributions that aren't wrappers for <random>
// ````````````````````````````````````````````````````````````````````````````````````````````````
// dirichlet
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(dirichlet_random_vector_should_be_right_size){
    static baxcat::PRNG rng(10);

    std::vector<double> alpha_1 = {1,2,1,.5};
    std::vector<double> p_1 = rng.dirrand(alpha_1);
    BOOST_CHECK_EQUAL( alpha_1.size(), p_1.size() );

    std::vector<double> alpha_2 = {1};
    std::vector<double> p_2 = rng.dirrand(alpha_2);
    BOOST_CHECK_EQUAL( alpha_2.size(), p_2.size() );
}

BOOST_AUTO_TEST_CASE(dirichlet_random_vector_should_be_normalized){
    static baxcat::PRNG rng(10);

    std::vector<double> alpha_1 = {1,2,1,.5};
    std::vector<double> p_1 = rng.dirrand(alpha_1);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::utils::sum(p_1), 1.0, TOL);

    std::vector<double> alpha_2 = {.1,.2,.1,.5};
    std::vector<double> p_2 = rng.dirrand(alpha_2);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::utils::sum(p_2), 1.0, TOL);
}

// symmetric dirichlet
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(symmetric_dirichlet_random_vector_should_be_right_size){
    static baxcat::PRNG rng(10);

    size_t K_1 = 4;
    double alpha_1 = 1;
    std::vector<double> p_1 = rng.dirrand(K_1, alpha_1);
    BOOST_CHECK_EQUAL( K_1, p_1.size() );

    char K_2 = 4;
    double alpha_2 = 1;
    std::vector<double> p_2 = rng.dirrand(K_2, alpha_2);
    BOOST_CHECK_EQUAL( K_2, p_2.size() );

    char K_3 = 2;
    double alpha_3 = .5;
    std::vector<double> p_3 = rng.dirrand(K_3, alpha_3);
    BOOST_CHECK_EQUAL( K_3, p_3.size() );

}

BOOST_AUTO_TEST_CASE(symmetric_dirichlet_random_vector_should_be_normalized){
    static baxcat::PRNG rng(10);

    size_t K_1 = 4;
    double alpha_1 = 1;
    std::vector<double> p_1 = rng.dirrand(K_1, alpha_1);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::utils::sum(p_1), 1.0, TOL);

    char K_2 = 4;
    double alpha_2 = 10;
    std::vector<double> p_2 = rng.dirrand(K_2, alpha_2);
    BOOST_CHECK_CLOSE_FRACTION( baxcat::utils::sum(p_2), 1.0, TOL);    
}

// Stress test
// ````````````````````````````````````````````````````````````````````````````````````````````````
BOOST_AUTO_TEST_CASE(parallel_stress_test){
    // make sure that the RNG works in a parallel setting
    unsigned int max_threads = omp_get_max_threads();
    std::cout << "stress test with " << max_threads << " thread(s)." << std::endl;
    static baxcat::PRNG prng(1032);
    int num_tests = 1000;
    std::vector<unsigned int> X(num_tests, 0);
    std::vector<bool> thread_use(max_threads, false);
    // generate a bunch of random number in parallel
#pragma omp parallel for shared(prng)
    for (int i = 0; i < num_tests; i++){
        int t_idx = omp_get_thread_num();
        unsigned int x = prng.randuint(1000)+1;
        X[i] = x;
        thread_use[t_idx] = true;
    }

    // make sure that omp used all threads
    for( bool thread_used: thread_use )
        BOOST_CHECK(thread_used);

    // make sure that none of the values are 0
    for( unsigned int x: X)
        BOOST_CHECK( x != 0 );
}

BOOST_AUTO_TEST_SUITE_END()
