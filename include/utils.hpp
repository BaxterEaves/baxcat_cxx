
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

#ifndef baxcat_cxx_utils_hpp
#define baxcat_cxx_utils_hpp

#include <vector>
#include <limits>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <algorithm>
#include <iostream>

#include "template_helpers.hpp"

namespace baxcat { namespace utils {


    // if key is not found, returns where it should have been
    static size_t binary_search( std::vector<size_t> A, size_t key )
    {
        int imin = 0;
        int imax = int(A.size()-1);
        int imid = 0;
        while (imax >= imin)
        {
            imid = imin+(imax-imin)/2;
            if(A[imid] == key){
                return imid;
            }else if (A[imid] < key){
                imin = imid + 1;
            }else{
                imax = imid - 1;
            }
        }
        return (A[imid] < key) ? imid + 1 : imid;
    }


    // Returns an n-length vector with uniform spacing from a to b. That is
    // vec[0] = a and vec[n-1] = b.
    static std::vector<double> linspace(double a, double b, unsigned int n)
    {
        assert(a<b);
        assert(n > 0);

        std::vector<double> ret(n,a);
        double interval = (b-a)/double(n-1);
        for(unsigned int i = 1; i < n; i++){
            ret[i] = a + i*interval;
        }
        return ret;
    };


    // returns an n-length vector with log spacing from a to b. Both a and b
    // must be positive.
    static std::vector<double> log_linspace(double a, double b, unsigned int n)
    {
        assert(a>=0);
        assert(a<b);
        assert(n > 0);

        // protct from log domain error
        double log_a = (a==0) ? -DBL_MAX/2 : log(a);
        double log_b = log(b);
        std::vector<double> ret = linspace(log_a, log_b, n);
        for(unsigned int i = 0; i < n; i++){
            ret[i] = exp(ret[i]);
        }

        return ret;
    };


    // returns the minimum element of v
    template <typename T>
    static baxcat::enable_if<std::is_arithmetic<T>, size_t> argmax(const std::vector<T> &v)
    {
        auto max = std::numeric_limits<T>::min();
        size_t index = 0;
        size_t i=0;
        for( T x : v){
            if( !std::isnan(x) and !std::isinf(x)){
                if(x > max){
                    max = x;
                    index = i;
                }
            }
            ++i;
        }
        return index;
    }


    // returns the minimum element of v
    template <typename T>
    static baxcat::enable_if<std::is_arithmetic<T>, T> vector_min(const std::vector<T> &v)
    {
        auto min = std::numeric_limits<T>::max();
        for( T x : v){
            if( !std::isnan(x) and !std::isinf(x)){
                if(x < min)
                    min = x;
            }
        }
        return min;
    }


    // returns the minimum element of v
    template <typename T>
    static baxcat::enable_if<std::is_arithmetic<T>, T> vector_max(const std::vector<T> &v)
    {
        auto max = std::numeric_limits<T>::min();
        for( T x : v){
            if( !std::isnan(x) and !std::isinf(x)){
                if(x > max)
                    max = x;
            }
        }
        return max;
    }


    // returns the mean of the vector v
    static double vector_mean(const std::vector<double> &v)
    {
        double sum = 0;
        double count = 0;
        for(double x : v){
            if( !std::isnan(x) and !std::isinf(x) ){
                sum += x;
                ++count;
            }
        }
        return sum/count;
    }


    // returns the sum of squared error of v
    static double sum_of_squares(const std::vector<double> &v)
    {
        double mu = vector_mean(v);
        double ss = 0;
        for(auto x : v){
            if(!std::isnan(x) and !std::isinf(x)){
                ss += (x-mu)*(x-mu);
            }
        }
        return ss;
    }


    template <typename T>
    static baxcat::enable_if<std::is_arithmetic<T>, T> sum( const std::vector<T> &V)
    {
        T ret = 0;
        for( T v : V){
            if(!std::isnan(v) and !std::isinf(v))
                ret += v;
        }
        return ret;
    }


    // prints the vector to a single line
    template <typename T>
    static void print_vector(const std::vector<T> &v){
        std::cout << "[";
        for(T x: v)
            std::cout << x << " ";
        std::cout << "]" << std::endl;
    }


    // prints the vector to a single line
    template <typename T>
    static void print_2d_vector(const std::vector<std::vector<T>> &v)
    {
        std::cout << "[\n";
        for(std::vector<T> x: v)
            print_vector(x);
        std::cout << "]" << std::endl;
    }

}}

#endif
