
#ifndef baxcat_cxx_feature_guard
#define baxcat_cxx_feature_guard

#include <map>
#include <vector>
#include <iostream>
#include <cmath>

#include "container.hpp"
#include "component.hpp"
#include "prng.hpp"
#include "utils.hpp"

// Pure abstract class so we can store a vector of different feature types
// in the View and State
// TODO: Store and update the feature logp (less crp value) for every transition
namespace baxcat {

class BaseFeature{
public:
    virtual std::shared_ptr<BaseFeature> clone() const = 0;

    // insert X[row] into cluster
    virtual void insertElement(size_t row, size_t cluster) = 0;
    virtual void insertElementToSingleton(size_t row) = 0;
    virtual void removeElement(size_t row, size_t cluster) = 0;
    // cast value and insert into cluster
    virtual void insertValue(double value, size_t cluster) = 0;
    virtual void removeValue(double value, size_t cluster) = 0;

    // updates the hyperparameters for each cluster
    virtual void updateHypers() = 0;

    // logp of the element in row in cluster
    virtual double elementLogp(size_t row, size_t cluster) const = 0;
    // logp of a specific value
    virtual double valueLogp(double value, size_t cluster) const = 0;
    // log p of the element in row in its own cluster
    virtual double singletonLogp(size_t row) const = 0;
    // log p of the element x in its own cluster
    virtual double singletonValueLogp(double value) const = 0;
    // the marginal/likelihoos of cluster
    virtual double clusterLogp(size_t cluster) const = 0;
    // the product of cluster_logp's
    virtual double logp() const = 0;
    // get the feature score
    virtual double logScore() const = 0;

    // move X[row] from cluster[move_from] to cluster[move_to]
    virtual void moveToCluster(size_t row, size_t move_from, size_t move_to) = 0;
    // destroy X[row]'s singleton cluster, cluster[to_destroy] and move it
    // to cluster[move_to]
    virtual void destroySingletonCluster(size_t row, size_t to_destroy, size_t move_to) = 0;
    // remove X[row] from cluster[current] and create a singleton
    virtual void createSingletonCluster(size_t row, size_t current) = 0;
    // delete all clusters and reassing X according to Z
    virtual void reassign(std::vector<size_t> assignment) = 0;

    // getters
    // returns the feature index
    virtual size_t getIndex() const = 0;
    // get the number of clusters
    virtual size_t getNumClusters() const = 0;
    // returns the number of rows
    virtual size_t getN() const = 0;
    // returns the hypers vector
    virtual std::vector<double> getHypers() const = 0;
    // returns a vector of maps containing the sufficient statistics of each
    // model in clusters
    virtual std::vector<std::map<std::string, double>> getModelSuffstats() const = 0;
    // returns a vector of maps containing the hyperparameters of each model in
    // clusters
    virtual std::vector<std::map<std::string, double>> getModelHypers() const = 0;
    virtual std::map<std::string, double> getHypersMap() const = 0;
    // get the set (not missing) data
    virtual std::vector<double> getData() const = 0;
    virtual double getDataAt(size_t row_index) const = 0;

    // setters
    // remove all clusters
    virtual void clear() = 0;
    // sets the hypers of all clusters in a feature with a string-indexed map
    virtual void setHypers(std::map<std::string, double> hypers_map) = 0;
    // sets the hypers of all clusters in a feature with a vector
    virtual void setHypers(std::vector<double> hypers_vec) = 0;
    // sets the hyperprior config
    virtual void setHyperConfig(std::vector<double> hyperprior_config) = 0;
    // cast and append datum to last row in a singleton cluster
    virtual void appendRow(double datum) = 0;
    // pop the last dataum
    virtual void popRow(size_t cluster_assignment) = 0;
    // // pop the singleton cluster
    // virtual void popSingleton(size_t cluster_index) = 0;

    // draw
    // draw from cluster
    virtual double drawFromCluster(size_t cluster_idx, baxcat::PRNG *rng) = 0;

    // replace value in data
    virtual void replaceValue(size_t which_row, size_t which_cluster, double x) = 0;

    // testing
    // resamples the data in the row
    virtual void __geweke_resampleRow(size_t which_row, size_t which_category, baxcat::PRNG *rng) = 0;
    virtual void __geweke_clear() = 0;
    virtual void __geweke_initHypers() = 0;

protected:
    size_t _N;

};


template <class DataType, typename T>
class Feature : public BaseFeature
{
private:
        size_t _index;
        baxcat::PRNG *_rng;
        std::vector<double> _distargs;
        std::vector<double> _hypers;
        std::vector<double> _hyperprior_config;
        baxcat::DataContainer<T> _data;
        std::vector<DataType> _clusters;

public:

    virtual std::shared_ptr<BaseFeature> clone() const {
        return std::shared_ptr<BaseFeature>(new Feature(static_cast<Feature const &>(*this)));
    };

    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        baxcat::PRNG *rng_ptr);
    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        std::vector<size_t> assignment, baxcat::PRNG *rng_ptr);
    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        baxcat::PRNG *rng_ptr, std::vector<double> hypers, std::vector<double> hyperprior_config);

    virtual void insertElement(size_t row, size_t cluster) final;
    virtual void insertElementToSingleton(size_t row) final;
    virtual void removeElement(size_t row, size_t cluster) final;
    virtual void insertValue(double value, size_t cluster) final;
    virtual void removeValue(double value, size_t cluster) final;

    virtual void updateHypers() override;

    virtual double elementLogp(size_t row, size_t cluster) const final;
    virtual double singletonLogp(size_t row) const final;
    virtual double valueLogp(double value, size_t cluster) const final;
    virtual double singletonValueLogp(double value) const final;
    virtual double clusterLogp(size_t cluster) const final;
    virtual double logp() const final;
    virtual double logScore() const final;

    virtual void moveToCluster(size_t row, size_t move_from, size_t move_to) final;
    virtual void destroySingletonCluster(size_t row, size_t to_destroy, size_t move_to) final;
    virtual void createSingletonCluster(size_t row, size_t current) override;
    virtual void reassign(std::vector<size_t> assignment) override;

    virtual size_t getIndex() const final;
    virtual size_t getN() const final;
    virtual size_t getNumClusters() const final;
    virtual std::vector<double> getHypers() const final;
    virtual std::vector<std::map<std::string, double>> getModelSuffstats() const final;
    virtual std::vector<std::map<std::string, double>> getModelHypers() const final;
    virtual std::map<std::string, double> getHypersMap() const final;
    virtual std::vector<double> getData() const final;
    virtual double getDataAt(size_t row_index) const final;

    virtual void clear() final;
    virtual void setHypers(std::map<std::string, double> hypers_map) final;
    virtual void setHypers(std::vector<double> hypers_vec) final;
    virtual void setHyperConfig(std::vector<double> hyperprior_config) final;
    virtual void appendRow(double datum) final;
    virtual void popRow(size_t cluster_assignment) final;

    virtual void replaceValue(size_t which_row, size_t which_cluster, double x) final;

    virtual double drawFromCluster(size_t cluster_idx, baxcat::PRNG *rng) final;

    virtual void __geweke_resampleRow(size_t which_row, size_t which_category, baxcat::PRNG *rng) final;
    virtual void __geweke_clear() final;
    virtual void __geweke_initHypers() final;
};

#include "feature.tpp"

}

#endif
