
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
    // insert X[row] into cluster
    virtual void insertElement(size_t row, size_t cluster) = 0;
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
    virtual std::vector<double> getData() const = 0;  // implement

    // setters
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

    // testing
    // resamples the data in the row
    virtual void __geweke_resampleRow(size_t which_row, size_t which_category, baxcat::PRNG *rng) = 0;
    virtual void __geweke_clear() = 0;
    virtual void __geweke_initHypers() = 0;

};


template <class DataType, typename T>
class Feature : public BaseFeature
{
private:
        size_t _index;
        baxcat::DataContainer<T> _data;
        baxcat::PRNG *_rng;
        size_t _N;
        std::vector<double> _distargs;
        std::vector<DataType> _clusters;
        std::vector<double> _hypers;
        std::vector<double> _hyperprior_config;

public:
    
    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        baxcat::PRNG *rng_ptr);
    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        std::vector<size_t> assignment, baxcat::PRNG *rng_ptr);
    Feature(unsigned int index, baxcat::DataContainer<T> data, std::vector<double> distargs,
        baxcat::PRNG *rng_ptr, std::vector<double> hypers, std::vector<double> hyperprior_config);

    virtual void insertElement(size_t row, size_t cluster) final;
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

    virtual void moveToCluster(size_t row, size_t move_from, size_t move_to) final;
    virtual void destroySingletonCluster(size_t row, size_t to_destroy, size_t move_to) final;
    virtual void createSingletonCluster(size_t row, size_t current) override;
    virtual void reassign(std::vector<size_t> assignment) override;

    virtual size_t getIndex() const final;
    virtual size_t getN() const final;
    virtual std::vector<double> getHypers() const final;
    virtual std::vector<std::map<std::string, double>> getModelSuffstats() const final;
    virtual std::vector<std::map<std::string, double>> getModelHypers() const final;
    virtual std::map<std::string, double> getHypersMap() const final;
    virtual std::vector<double> getData() const final;

    virtual void setHypers(std::map<std::string, double> hypers_map) final;
    virtual void setHypers(std::vector<double> hypers_vec) final;
    virtual void setHyperConfig(std::vector<double> hyperprior_config) final;
    virtual void appendRow(double datum) final;
    virtual void popRow(size_t cluster_assignment) final;

    virtual double drawFromCluster(size_t cluster_idx, baxcat::PRNG *rng) final;

    virtual void __geweke_resampleRow(size_t which_row, size_t which_category, baxcat::PRNG *rng) final;
    virtual void __geweke_clear() final;
    virtual void __geweke_initHypers() final;
};

#include "feature.tpp"

}

#endif
