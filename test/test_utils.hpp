
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

#ifndef baxcat_cxx_test_utils
#define baxcat_cxx_test_utils

#include <mgl2/mgl.h> // MathGL plotting
#include <algorithm>
#include <vector>
#include <cmath>
#include <sstream>

#include "prng.hpp"
#include "state.hpp"
#include "plotting.hpp"
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
    static typename baxcat::enable_if<std::is_integral<T>,double> chi2gof(std::vector<T> X, 
                                                                          std::vector<T> Y)
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
    static typename baxcat::enable_if<std::is_integral<T>,double> chi2Stat(std::vector<T> X)
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

    template <typename lambda>
    static double oneSampleKSTest(std::vector<double> X, lambda cdf, bool do_plot=false,
        mglGraph *gr=nullptr, std::string function_name="f(x)")
    {
        double n = double(X.size());

        std::sort(X.begin(), X.end());
        std::vector<double> sample_cdf = baxcat::utils::linspace(1.0/n,1.0,X.size());
        std::vector<double> true_cdf_vector(n,0);

        double ks_stat = 0;
        for(size_t i = 0; i < X.size(); ++i){
            double sample_cdf_val = sample_cdf[i];
            double true_cdf_val = cdf(X[i]);
            double dist = fabs(sample_cdf_val - true_cdf_val);
            if( dist > ks_stat )
                ks_stat = dist;
            true_cdf_vector[i] = true_cdf_val;
        }

        if(do_plot and gr)
            baxcat::plotting::pPPlot(gr, true_cdf_vector, sample_cdf, function_name, "true",
                "sample");

        return ks_stat;
    }

    static double twoSampleKSTest(std::vector<double> a, std::vector<double> b, bool do_plot=false,
        mglGraph *gr=nullptr, std::string function_name=" ")
    {
        std::sort(a.begin(), a.end());
        std::sort(b.begin(), b.end());

        std::vector<double> X = a;
        X.reserve(X.size() + distance(b.begin(),b.end()));
        X.insert(X.end(),b.begin(),b.end());

        std::sort(X.begin(), X.end());

        // for potting
        std::vector<double> cdf_a(X.size());
        std::vector<double> cdf_b(X.size());

        double F_a = 0;
        double F_b = 0;

        size_t index_a = 0;
        size_t index_b = 0;
        double n_b = double(b.size());
        double n_a = double(a.size());

        double ks_stat = 0;

        size_t i = 0;
        for(auto x : X){
            if(a[index_a] < x){
                ++index_a;
                F_a = double(index_a)/n_a;
            }

            if(b[index_b] < x){
                ++index_b;
                F_b = double(index_b)/n_b;
            }

            double dist = fabs(F_a-F_b);
            if (dist > ks_stat)
                ks_stat = dist;

            cdf_a[i] = F_a;
            cdf_b[i] = F_b;

            ++i;
        }

        if(do_plot and gr)
            baxcat::plotting::pPPlot(gr, cdf_a, cdf_b, function_name, "sample a", "sample b");

        return ks_stat;

    }

    // reject null hypothesis at p = .05
    static bool ksTestRejectNull(double ks_stat, size_t n_a, size_t n_b){
        double na = double(n_a);
        double nb = double(n_b);
        return ks_stat > 1.36 * sqrt((na+nb)/(na*nb));
    }

    // test slice sampler
    template <typename lambda_pdf, typename lambda_cdf>
    static double testSliceSampler(double x_0, lambda_pdf &log_pdf, baxcat::Domain sampler_domain,
        double w, lambda_cdf &cdf, size_t n_samples=500, size_t lag=25)
    {
        static baxcat::PRNG *prng = new baxcat::PRNG();

        std::vector<double> samples(n_samples,0);

        for( auto &x : samples ){
            x_0 = baxcat::samplers::sliceSample(x_0, log_pdf, sampler_domain, w, lag, prng);
            x = x_0;
        }

        double ks_stat = oneSampleKSTest(samples, cdf, false, nullptr);
        return ks_stat;
    }

    template <typename lambda>
    static double testHyperparameterSampler(const lambda &log_f, double x_0,
        baxcat::Domain sampler_domain, double w, std::string filename=" ",
        std::string function_name="f(x)", size_t n_samples=1000, size_t lag=50)
    {

        static baxcat::PRNG *prng = new baxcat::PRNG();
        std::vector<double> samples(n_samples,0);

        for( auto &x : samples ){
            x_0 = baxcat::samplers::sliceSample(x_0, log_f, sampler_domain, w, lag, prng);
            x = x_0;
        }

        std::sort(samples.begin(), samples.end());

        mglGraph gr;
        gr.SetSize(1000,1000);

        size_t n_bins = 25; // histogram bins

        gr.SubPlot(2,2,0);
        baxcat::plotting::hist(&gr, samples, n_bins, function_name);


        double a,b;
        if(sampler_domain.lower != ALMOST_ZERO)
            a = baxcat::utils::vector_min(samples);
        else
            a = sampler_domain.lower;

        b = baxcat::utils::vector_max(samples);

        const auto exp_f = [log_f](double x){ return exp(log_f(x)); };

        gr.SubPlot(2,2,1);
        baxcat::plotting::functionPlot(&gr, samples, exp_f, function_name, "x", "f(x)");

        double z_f = baxcat::numerics::quadrature(exp_f, a, b);
        const auto cdf = [exp_f, z_f, a](double x){return baxcat::numerics::quadrature(exp_f, a, x)/z_f;};

        gr.SubPlot(2,2,2);
        double ks_stat = oneSampleKSTest(samples, cdf, true, &gr, function_name);

        // plot true cdf values
        gr.SubPlot(2,2,3);
        baxcat::plotting::functionPlot(&gr, samples, cdf, "cdf", "x", "");

        gr.WriteFrame(filename.c_str());

        return ks_stat;
    }

    static void __output_ks_test_result(bool reject_null, double ks_stat, std::string test_name)
    {
        std::string pass_message = reject_null ? " __FAIL__" : " __PASS__";
        std::cout << "KS-statistic (" << test_name << "): " << ks_stat;
        std::cout << pass_message << std::endl;
    }

    static void __update_pass_counters(size_t &num_pass, size_t &num_fail, bool &all_pass, bool test_pass)
    {
        if(test_pass){
            ++num_pass;
        }else{
            ++num_fail;
            all_pass = false;
        }
    }


    static bool testOneDimInfereneceQuality(size_t num_rows, size_t num_clusters, size_t num_transitions,
        double separation, std::string datatype, std::string filename)
    {
        std::vector<double> view_weights = {1};
        std::vector<double> weights(num_clusters, 1.0/double(num_clusters));
        std::vector<std::vector<double>> category_weights(1,weights);
        std::vector<double> category_separation = {separation};
        std::vector<string> datatypes = {datatype};
        size_t seed = 10;

        auto datatype_conv = getDatatypes({datatype})[0];

        // Note: 5 is the number of categorical categories used by synthetic data generator.
        size_t multinomial_k = 5;

        // TODO: Valid distargs for all types
        std::vector<std::vector<double>> distargs;
        if(datatype_conv == baxcat::datatype::continuous){
            distargs = {{0}};
        }else if(datatype_conv == baxcat::datatype::categorical){
            distargs = {{static_cast<double>(multinomial_k)}};
        }

        baxcat::SyntheticDataGenerator sdg(num_rows, view_weights, category_weights,
                                           category_separation, datatypes, seed);

        auto x_orig = sdg.getData();
        
        baxcat::State state(x_orig, datatypes, distargs, 0);
        state.transition({},{},{},0,num_transitions);

        // FIXME: when unobserved sample is implemented, use unobserved query
        std::vector<std::vector<size_t>> query(num_rows);
        for(size_t i = 0; i < num_rows; ++i)
            query[i] = {i,0};

        auto x_predict_raw = state.predictiveDraw(query,{},{},num_rows);

        std::vector<double> x_predict(num_rows);
        for(size_t i = 0; i < num_rows; ++i)
            x_predict[i] = x_predict_raw[0][i];

        std::stringstream ss;
        ss << num_clusters << "-cluster, 1-columns " << datatype << " inference (sep=";
        ss << separation << ")";

        // If categorical, compile into counts and compute expected counts from SDG

        if(datatype_conv == baxcat::datatype::categorical){
            std::vector<double> original_data(multinomial_k, 0);
            std::vector<double> counts_observed(multinomial_k, 0);
            for(auto &x : x_orig[0]){
                size_t idx = static_cast<size_t>(x + .5);
                ++original_data[idx];
            }
            for(auto &x : x_predict){
                size_t idx = static_cast<size_t>(x + .5);
                ++counts_observed[idx];
            }

            double scale = static_cast<double>(num_rows);

            std::vector<double> f_true(multinomial_k, 0);
            std::vector<double> f_inferred(multinomial_k, 0);
            std::vector<double> counts_expected(multinomial_k, 0);

            for(size_t k = 0; k < multinomial_k; ++k){
                double x = static_cast<double>(k);
                f_true[k] = exp(sdg.logLikelihood({x}, 0)[0]);
                counts_expected[k] = f_true[k]*scale;
                auto fx = state.predictiveLogp({{num_rows, 0}}, {x}, {}, {});
                f_inferred[k] = exp(fx[0]);
            }

            ASSERT(std::cout, fabs(baxcat::utils::sum(counts_expected)-scale) < .1);
            ASSERT(std::cout, fabs(baxcat::utils::sum(counts_observed)-scale) < .1);

            // uses observed counts rather than expected counts because the observed counts can be
            // arr often different enough from the true distribution to throw off the Chi-square 
            // test
            double chi_sqared_stat = chi2Stat(counts_observed, original_data);
            double df = static_cast<double>(multinomial_k)-1;
            bool distributions_differ = (1-numerics::rgamma(chi_sqared_stat/2, df/2)) < .05;

            mglGraph gr;
            gr.SetSize(1000, 1000);

            gr.SubPlot(2, 2, 0);
            baxcat::plotting::hist(&gr, f_true, multinomial_k, "True distribution", true);

            gr.SubPlot(2, 2, 1);
            baxcat::plotting::hist(&gr, f_inferred, multinomial_k, "Inferred distribution", true);

            gr.SubPlot(2, 2, 2);
            baxcat::plotting::hist(&gr, original_data, multinomial_k, "Original Data", true);

            gr.SubPlot(2, 2, 3);
            baxcat::plotting::hist(&gr, counts_observed, multinomial_k, "Sampled Data", true);
            gr.WriteFrame(filename.c_str());

            return distributions_differ;

        }else{
            // plot results
            mglGraph gr;
            gr.SetSize(1000, 1000);
            gr.SubPlot(2, 2, 0);
            double ks_stat = baxcat::test_utils::twoSampleKSTest(x_orig[0], x_predict, true, &gr,
                ss.str());

            gr.SubPlot(2, 2, 1);
            baxcat::plotting::hist(&gr, x_orig[0], 30, "Original Data");

            gr.SubPlot(2, 2, 2);
            baxcat::plotting::hist(&gr, x_predict, 30, "Sampled Data");

            // get range for pdf plotting
            auto x_min = baxcat::utils::vector_min(x_orig[0]);
            auto x_max = baxcat::utils::vector_max(x_orig[0]);

            auto X = baxcat::utils::linspace(x_min, x_max, 200);

            // get pdfs
            gr.SubPlot(2, 2, 3);
            auto f_orignal = sdg.logLikelihood(X, 0);
            for( auto &f : f_orignal)
                f = exp(f);

            std::vector<double> f_inferred;
            for (auto x : X){
                auto fx = state.predictiveLogp({{num_rows,0}}, {x}, {}, {});
                f_inferred.push_back( exp(fx[0]) );
            }

            baxcat::plotting::compPlot(&gr, X, f_orignal, f_inferred, "original vs inferred pdf");

            bool distributions_differ = baxcat::test_utils::ksTestRejectNull(ks_stat, num_rows, num_rows);

            // __output_ks_test_result(distributions_differ, ks_stat, ss.str());

            gr.WriteFrame(filename.c_str());

            return distributions_differ;
        }

        

        
    }

}} // end namespaces

#endif
