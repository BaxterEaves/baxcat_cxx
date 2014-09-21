
#ifndef baxcat_cxx_component
#define baxcat_cxx_component

#include "debug.hpp"

#include <map>
#include <vector>
#include <string>

#include "prng.hpp"
#include "utils.hpp"

template <typename T>
class Component{
public:
    // Component() : n_(0){};
    // Component(double n) : n_(n)c{};

    // insert element x into the sufficient statistics
    virtual void insertElement(T x) = 0;
    // remove element x from the sufficient statistics
    virtual void removeElement(T x) = 0;
    // clear sufficient statistics
    virtual void clear(const std::vector<double> &distargs) = 0;

    // get a vector of the hyperparameters
    virtual std::vector<double> getHypers() const = 0;
    // set hypers with vector
    virtual void setHypers(std::vector<double> hypers) = 0;
    // set hyper with string-double map. used during feature intialization
    virtual void setHypersByMap( std::map<std::string, double> hypers_map) = 0;

    // returns a string-indexed map of the hyperparmeters
    virtual std::map<std::string, double> getHypersMap() const = 0;
    // returns a string-indexed map of the sufficient statistics
    virtual std::map<std::string, double> getSuffstatsMap() const = 0;
    
    // probabilities
    // marginal/likelihood of the data currently assigned
    virtual double logp() const = 0;
    // the predicitve/likelihood of x in this model
    virtual double elementLogp(T x) const = 0;
    // the predictive probability of x in its own model
    virtual double singletonLogp(T x) const = 0;
    // prior probability of hyperparameters
    virtual double hyperpriorLogp(const std::vector<double> &hyperprior_config) const = 0;

    // drawing elements (unconstrained)
    // draw an element from the model
    virtual T draw(baxcat::PRNG *rng) const = 0;
    // draw an element for the model given that the data in constraints
    // are also assigned to the model
    virtual T drawConstrained(std::vector<T> constraints, baxcat::PRNG *rng) const = 0;
    
    size_t getCount(){ return static_cast<size_t>(_n+.5); };
protected:
    // Number of data points assigned to the model
    double _n;
};


template <class ModelType, typename T>
class SubComponent : public Component<T>{
public:

    // construct the hyperprior config
    static std::vector<double> constructHyperpriorConfig( const std::vector<T> &X )
    {
        return ModelType::constructHyperpriorConfig(X);
    };
    
    // draw hyperparameters from hyperprior
    static std::vector<double> initHypers( const std::vector<double> &hyperprior_config, 
        baxcat::PRNG *rng )
    {
        return ModelType::initHypers( hyperprior_config, rng );
    }
    
    // resample hyperparameters from posterior
    static std::vector<double> resampleHypers( std::vector<ModelType> &models,
        const std::vector<double> &hyperprior_config, baxcat::PRNG *rng)
    {
        return ModelType::resampleHypers(models, hyperprior_config, rng);
    }
    
};

#endif
