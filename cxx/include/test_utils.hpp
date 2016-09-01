
#ifndef baxcat_cxx_test_utils
#define baxcat_cxx_test_utils

/* #include <mgl2/mgl.h> // MathGL plotting */
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>

#include "prng.hpp"
#include "state.hpp"
/* #include "plotting.hpp" */
#include "utils.hpp"
#include "debug.hpp"
#include "samplers/slice.hpp"
#include "helpers/constants.hpp"
#include "helpers/state_helper.hpp"
#include "helpers/synthetic_data_generator.hpp"

using baxcat::helpers::getDatatypes;

namespace baxcat{
namespace test_utils{


    static double chi2Stat(std::vector<double> observed, std::vector<double> expected)
    {
        double correction = 10E-6;
        double stat = 0;
        for( size_t i = 0; i < observed.size(); ++i){ 
            // avoids divide by zero error
            if(expected[i] == 0){
                observed[i] += correction;
                expected[i] += correction;
            }
            stat += (observed[i]-expected[i])*(observed[i]-expected[i])/expected[i];    
        }
        return stat;
    }


    template <typename T>
    static typename baxcat::enable_if<std::is_integral<T>,double> chi2gof(
            std::vector<T> X, std::vector<T> Y)
    {
        double N_X = static_cast<double>(X.size());
        double N_Y = static_cast<double>(Y.size());
        size_t num_bins_x = baxcat::utils::vector_max(X)+1;
        size_t num_bins_y = baxcat::utils::vector_max(Y)+1;

        size_t num_bins = num_bins_x > num_bins_y ? num_bins_x : num_bins_y;

        std::vector<double> x_count(num_bins);
        std::vector<double> y_count(num_bins);

        for(T &x : X)
            ++x_count[x];

        for(T &y : Y)
            ++y_count[y];

        for(auto &c : x_count)
            c /= N_X;

        for(auto &c : y_count)
            c /= N_Y;

        return chi2Stat(x_count, y_count);
    }


    template <typename T>
    static typename baxcat::enable_if<std::is_integral<T>,double> chi2Stat(
            std::vector<T> X)
    {
        // null hypothesis is uniform
        double N = static_cast<double>(X.size());
        size_t num_bins = baxcat::utils::vector_max(X)+1;
        std::vector<double> observed(num_bins);
        std::vector<double> expected(num_bins, N/static_cast<double>(num_bins));
        for( T &x : X)
            ++observed[x];

        return chi2Stat(observed, expected);
    }


    static double chi2Stat(std::vector<double> X, size_t num_bins=0)
    {
        // null hypothesis is uniform
        double N = static_cast<double>(X.size());

        if(num_bins == 0)
            num_bins = baxcat::utils::vector_max(X)+1;

        std::vector<double> observed(num_bins);
        std::vector<double> expected(num_bins, N/static_cast<double>(num_bins));
        for( double &x : X)
            ++observed[static_cast<size_t>(x+.5)];

        return chi2Stat(observed, expected);
    }


    // returns -1 is a is empty, 0 if element was not found, 1 if element found
    template <typename T>
    static int hasElement(std::vector<T> a, T element)
    {
        if(a.empty())
            return -1;

        if(std::find(a.begin(), a.end(), element) != a.end()){
            return 1;
        }else{
            return 0;
        }
    }

    // returns -1 if a and be are differnt size, 1 if identical, and 0 if a and
    // b are not identical.
    template <typename T>
    static int areIdentical(std::vector<T> a, std::vector<T> b)
    {
        size_t sa = a.size();
        size_t sb = b.size();
        if(sa != sb || sb == 0 )
            return -1;

        for(unsigned int i = 0; i < a.size(); i++){
            if( a[i] != b[i] )
                return 0;
        }

        return 1;
    }


    // checks if a and b have the same elements regardless of order
    // returns 1 if true, 0 if false, and -1 if a and b are not the same size
    template <typename T>
    static int hasSameElements(std::vector<T> a, std::vector<T> b)
    {
        int ret = areIdentical(a, b);
        if(ret == 0 ){
            for(unsigned int i = 0; i < a.size(); i++){
                for(unsigned int j = 0; j < b.size(); j++){
                    if( a[i] == b[j] ){
                        b.erase(b.begin()+j);
                        break;
                    }
                }
            }
            return (b.size() == 0) ? 1 : 0;
        }else{
            return ret;
        }
    }


    // reject null hypothesis at p = .05
    static bool ksTestRejectNull(double ks_stat, size_t n_a, size_t n_b){
        double na = double(n_a);
        double nb = double(n_b);
        return ks_stat > 1.36 * sqrt((na+nb)/(na*nb));
    }

    // test slice sampler
    template <typename lambda_pdf, typename lambda_cdf>
    static double testSliceSampler(double x_0, lambda_pdf &log_pdf,
            baxcat::Domain sampler_domain, double w, lambda_cdf &cdf,
            size_t n_samples=500, size_t lag=25)
    {
        static baxcat::PRNG *prng = new baxcat::PRNG();

        std::vector<double> samples(n_samples,0);

        for( auto &x : samples ){
            x_0 = baxcat::samplers::sliceSample(x_0, log_pdf, sampler_domain,
                                                w, lag, prng);
            x = x_0;
        }

        double ks_stat = oneSampleKSTest(samples, cdf, false, nullptr);
        return ks_stat;
    }


    static void __output_ks_test_result(bool reject_null, double ks_stat,
            std::string test_name)
    {
        std::string pass_message = reject_null ? " __FAIL__" : " __PASS__";
        std::cout << "\tKS-statistic (" << test_name << "): " << ks_stat;
        std::cout << pass_message << std::endl;
    }


    static void __update_pass_counters(size_t &num_pass, size_t &num_fail,
            bool &all_pass, bool test_pass)
    {
        if(test_pass){
            ++num_pass;
        }else{
            ++num_fail;
            all_pass = false;
        }
    }


}} // end namespaces

#endif
