
#ifndef baxcat_cxx_parallel_rng_hpp
#define baxcat_cxx_parallel_rng_hpp

#define IndexError
#define NoMatchFound

#include <random>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "template_helpers.hpp"
#include "utils.hpp"
#include "numerics.hpp"
#include "omp.h"


// Parallel Random Number generator
// Super-cool ski instructor says: If you don't inialize this as static, you're
// gonna have a bad time.
namespace baxcat{
class PRNG{

    private:
        unsigned int num_threads;
        std::vector<std::mt19937> rngs;

    public:
        PRNG(unsigned int seed=0)
        {

            if(seed == 0){
                std::random_device rd;
                seed = rd();
            }

            unsigned int n_threads = omp_get_max_threads();
            // std::cout << "prng " << this << ": " << n_threads << " threads." << std::endl;

            // creata a PrallelRNG object with n_threads thread starting with
            // seed.
            num_threads = n_threads;

            // create an RNG to seed the RNGs in the vector
            std::mt19937 seeder(seed);
            std::uniform_int_distribution<unsigned int> urnd;

            // fill the vector with unique_ptrs to the RNGs
            for(unsigned int i = 0; i < n_threads; i++){
                unsigned int this_seed = urnd(seeder);
                rngs.emplace_back(this_seed);
            }
        };

        static void throwError(){
            throw std::logic_error("PRNG Error");
        }

        void seed(unsigned int new_seed)
        {
            if(new_seed == 0){
                std::random_device rd;
                new_seed = rd();
            }

            // create an RNG to seed the RNGs in the vector
            std::mt19937 seeder(new_seed);
            std::uniform_int_distribution<unsigned int> urnd;

            // fill the vector with unique_ptrs to the RNGs
            for(unsigned int i = 0; i < num_threads; i++){
                unsigned int this_seed = urnd(seeder);
                rngs[i].seed(this_seed);
            }
        }

        // get one of the rngs to use in some distribution
        std::mt19937& getRNG()
        {
            int idx = omp_get_thread_num();
            return rngs[idx];
        }

        // get one of the rngs to use in some distribution
        std::mt19937& getRNGByIndex(unsigned int index)
        {
            if( index > num_threads-1 ){
                std::cout << "getRNGByIndex: index out of bounds" << std::endl;
                throw std::logic_error("IndexError");
            }
            return rngs[index];
        }

        // return uniform random integer in [a,b)
        int randint(int a, int b)
        {
            std::uniform_int_distribution<int> dist(a, b-1);
            return dist(getRNG());
        }

        size_t randuint(size_t a)
        {
            std::uniform_int_distribution<size_t> dist(0, a-1);
            return dist(getRNG());
        }

        // return a random float in (0,1]
        double rand()
        {
            std::uniform_real_distribution<double> dist(0, 1);
            return dist(getRNG());
        }

        // return a random float in (a,b]
        double urand(double a, double b)
        {
            std::uniform_real_distribution<double> dist(a, b);
            return dist(getRNG());
        }

        // return a random element from the vector
        template<typename T>
        T randomElement( std::vector<T> X )
        {
            int index = randuint(X.size());
            return X[index];
        }

        // return a shuffled version of the vector
        template<typename T>
        std::vector<T> shuffle( std::vector<T> X)
        {
            size_t len = X.size();
            std::vector<T> ret;
            for( unsigned int i = 0; i < len; i++ ){
                assert( X.size() > 0);
                size_t index = randuint(X.size());
                ret.push_back(X[index]);
                X.erase(X.begin()+index);
            }
            return ret;
        }

        // multinomial draw from a vector of probabilities
        size_t pflip(std::vector<double> P)
        {
            // normalize
            double sum = 0;
            for( double p : P)
                sum += p;
            double r = rand();
            double cumsum = 0;
            for(size_t i = 0; i < P.size(); i++){
                cumsum += P[i]/sum;
                if( r < cumsum)
                    return i;
            }
            std::cout << "pflip: no match found" << std::endl;
            throw std::logic_error("NoMatchFound");
        }

        // multinomial draw from a vector of log probabilities
        size_t lpflip(std::vector<double> P)
        {
            // normalize
            double Z = baxcat::numerics::logsumexp(P);
            double r = rand();
            double cumsum = 0;
            for(size_t i = 0; i < P.size(); i++){
                cumsum += exp(P[i]-Z);
                if( r < cumsum)
                    return i;
            }

            baxcat::utils::print_vector(P);
            std::cout << "lpflip: no match found" << std::endl;
            throw std::logic_error("NoMatchFound");
        }

        // constructs a parition, Z, with K categories, and counts, Nk, from CRP(alpha)
        void crpGen(double alpha, size_t N, std::vector<size_t> &Z, size_t &K,
                    std::vector<size_t> &Nk)
        {
            // setup
            Z.resize(N);
            Nk = {1};
            K = 1;

            double log_alpha = log(alpha);
            double denom = alpha + 1;

            std::vector<double> logps(2, 0);

            for(size_t i = 1; i < N; ++i){
                double log_denom = log(denom);
                for( size_t k = 0; k < K; ++k)
                    logps[k] = log(double(Nk[k])) - log_denom;
                logps.back() = log_alpha - log_denom;

                size_t z = lpflip(logps);

                Z[i] = z;
                // if a new category has been added, add elements where needed
                if(z == K){
                    logps.push_back(0);
                    Nk.push_back(1);
                    ++K;
                }else{
                    ++Nk[z];
                }

                ++denom;

                ASSERT(std::cout, Nk.size() == K);
                if(fabs(denom - (i+1+alpha)) > 10e-10){
                    printf("CRPGen failure\n");
                    std::cout << "diff: " << fabs(denom - i+1+alpha) << std::endl;
                    std::cout << "denom: " << denom << std::endl;
                    std::cout << "i: " << i << std::endl;
                    std::cout << "alpha: " << alpha << std::endl;
                    std::cout << "N: " << N << std::endl;
                    std::cout << "z: " << z << std::endl;
                    std::cout << "K: " << K << std::endl;
                    assert(false);
                }
            }
        }

        // random distributions
        //`````````````````````````````````````````````````````````````````````````````````````````
        // gamma
        double gamrand(double shape, double scale)
        {
            ASSERT(std::cout, shape > 0);
            ASSERT(std::cout, scale > 0);

            std::gamma_distribution<double> dist(shape, scale);
            return dist(getRNG());
        }

        // inverse-gamma
        double invgamrand(double shape, double scale)
        {
            ASSERT(std::cout, shape > 0);
            ASSERT(std::cout, scale > 0);

            std::gamma_distribution<double> dist(shape, 1./scale);
            return 1./dist(getRNG());
        }

        // beta
        double betarand(double a, double b)
        {
            ASSERT(std::cout, a > 0);
            ASSERT(std::cout, b > 0);

            std::chi_squared_distribution<double> dist_a(2*a);
            std::chi_squared_distribution<double> dist_b(2*b);
            double za = dist_a(getRNG());
            double zb = dist_b(getRNG());
            return za/(za+zb);
        }

        // normal
        double normrand(double mu, double sigma)
        {
            ASSERT(std::cout, sigma > 0);

            std::normal_distribution<double> dist(mu, sigma);
            return dist(getRNG());
        }

        // student's t
        double trand(double nu)
        {
            ASSERT(std::cout, nu > 0);

            std::student_t_distribution<double> dist(nu);
            return dist(getRNG());
        }

        // lognormal
        double lognormrand(double mu, double sigma)
        {
            ASSERT(std::cout, sigma > 0);

            std::lognormal_distribution<double> dist(mu, sigma);
            return dist(getRNG());
        }

        // negative binomial
        size_t negbinrand(size_t r, double p)
        {
            ASSERT(std::cout, r > 0);
            ASSERT(std::cout, 0 < p and p < 1);

            std::negative_binomial_distribution<size_t> dist(r, p);
            return dist(getRNG());
        }

        // poisson
        size_t poissrand(double lambda)
        {
            ASSERT(std::cout, lambda > 0);

            std::poisson_distribution<size_t> dist(lambda);
            return dist(getRNG());
        }

        // dirichlet
        std::vector<double> dirrand(std::vector<double> alpha)
        {
            std::vector<double> P(alpha.size());
            double sum = 0;
            for(size_t i = 0; i < alpha.size(); ++i){
                P[i] = gamrand(alpha[i], 1);
                sum += P[i];
            }

            for(auto &p : P)
                p /= sum;

            return P;
        }

        // dirichlet (symmetric)
        template <typename T>
        typename baxcat::enable_if<std::is_integral<T>, std::vector<double>> dirrand(T K,
                                                                                     double alpha)
        {
            std::vector<double> P(K);
            double sum = 0;

            for(auto &p : P){
                p = gamrand(alpha, 1);
                sum += p;
            }

            for(auto &p : P)
                p /= sum;

            return P;
        }

        // Von Mises
        double vmrand(double mu, double kappa)
        {
            double a = 1 + sqrt(1 + 4 * (kappa*kappa));
            double b = (a - sqrt(2 * a))/(2 * kappa);
            double r = (1 + b*b)/(2 * b);
            double vmr;

            size_t MAX_ITERS = 100;
            size_t ITERS = 0;
            while (true) {
                double U1 = rand();
                double z = cos(M_PI * U1);
                double f = (1 + r * z)/(r + z);
                double c = kappa * (r - f);
                double U2 = rand();
                if (c * (2 - c) - U2 > 0){
                    double U3 = rand();
                    vmr = baxcat::numerics::sgn(U3 - 0.5) * acos(f) + mu;
                    vmr = fmod(vmr, 2.0*M_PI);
                    return vmr;
                }else if (log(c/U2) + 1 - c >= 0){
                    double U3 = rand();
                    vmr = baxcat::numerics::sgn(U3 - 0.5) * acos(f) + mu;
                    vmr = fmod(vmr, 2.0*M_PI);
                    return vmr;
                }

                if (ITERS > MAX_ITERS){
                    printf("vmrand(%f, %f) reached max iters.\n", mu, kappa);
                    return -1;
                }
            }
        }

};

} // end namespace baxcat

#endif
