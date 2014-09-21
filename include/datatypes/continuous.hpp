
#ifndef baxcat_cxx_continuous_hpp
#define baxcat_cxx_continuous_hpp

#include <map>
#include <cmath>
#include <string>
#include <vector>

#include "utils.hpp"
#include "numerics.hpp"
#include "samplers/slice.hpp"
#include "distributions/gamma.hpp"
#include "distributions/gaussian.hpp"
#include "distributions/students_t.hpp"
#include "distributions/inverse_gamma.hpp"

#include "component.hpp"
#include "models/nng.hpp"

namespace baxcat{
namespace datatypes{

class Continuous : public SubComponent<Continuous, double>{
public:

    Continuous(std::vector<double> &distargs) :
        _sum_x(0), _sum_x_sq(0), _m(0), _r(1), _s(1), _nu(1)
    {
        _n = 0;
        _log_Z0 = _nng.logZ(_r, _s, _nu);
        _log_ZN = _log_Z0;
    }

    Continuous(double n=0, double sum_x=0, double sum_x_sq=0, double m=0, double r=1,
               double s=1, double nu=1)
        : _sum_x(sum_x), _sum_x_sq(sum_x_sq) ,_m(m), _r(r), _s(s), _nu(nu)
    {
        _n = n;
        _log_Z0 = _nng.logZ(_r, _s, _nu);

        double mn = _m;
        double rn = _r;
        double sn = _s;
        double nun = _nu;

        _nng.posteriorParameters(_n, _sum_x, _sum_x_sq, mn, rn, sn, nun);
        _log_ZN = _nng.logZ(rn, sn, nun);;
    }

    // overrides
    // utilities
    virtual void insertElement(double x) override;
    virtual void removeElement(double x) override;
    virtual void clear(const std::vector<double> &distargs) override;

    virtual std::vector<double> getHypers() const override;
    virtual void setHypers(std::vector<double> hypers) override;
    virtual void setHypersByMap( std::map<std::string, double> hypers_map) override;

    virtual std::map<std::string, double> getHypersMap() const override;
    virtual std::map<std::string, double> getSuffstatsMap() const override;

    // probabilities
    virtual double logp() const override;
    virtual double elementLogp(double x) const override;
    virtual double singletonLogp(double x) const override;
    virtual double hyperpriorLogp(const std::vector<double> &hyperprior_config) const override;

    // draw
    virtual double draw(baxcat::PRNG *rng) const override;
    virtual double drawConstrained(std::vector<double> contraints,
        baxcat::PRNG *rng) const override;

    // hypers
    static std::vector<double> constructHyperpriorConfig( const std::vector<double> &X );

    static std::vector<double> initHypers( const std::vector<double> &hyperprior_config,
        baxcat::PRNG *rng );

    static std::vector<double> resampleHypers( std::vector<Continuous> &models,
        const std::vector<double> &hyperprior_config, baxcat::PRNG *rng, size_t burn=25);

    // construct hyper-parameter conditionals
    static std::function<double(double)> constructMConditional(
        const std::vector<Continuous> &models, const std::vector<double> &hyperprior_config);

    static std::function<double(double)> constructRConditional(
        const std::vector<Continuous> &models, const std::vector<double> &hyperprior_config);

    static std::function<double(double)> constructSConditional(
        const std::vector<Continuous> &models,const std::vector<double> &hyperprior_config);

    static std::function<double(double)> constructNuConditional(
        const std::vector<Continuous> &models);

    // updates normalizing constants
    void updateConstants();

protected:
    // hyperparameter conditionals
    double hyperMConditional_(double m) const;
    double hyperRConditional_(double r) const;
    double hyperSConditional_(double s) const;
    double hyperNuConditional_(double nu) const;

private:

    baxcat::models::NormalNormalGamma _nng;

    // for indexing
    enum hyper_idx {HYPER_M=0, HYPER_R=1, HYPER_S=2, HYPER_NU=3};
    enum hyperprior_config {M_MEAN=0, M_STD=1, R_SCALE=2, S_SCALE=3};

    // normalizing constants (user will need to specify when to update!)
    double _log_Z0;
    double _log_ZN;

    // sufficient statistics
    double _sum_x;
    double _sum_x_sq;

    // hyperparameters
    double _m;
    double _r;
    double _s;
    double _nu;
};

}} // end namespaces (baxcat::wrappers)

#endif
