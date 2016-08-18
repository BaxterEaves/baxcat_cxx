
#ifndef baxcat_cxx_view_guard
#define baxcat_cxx_view_guard

#include <vector>
#include <cmath>
#include <memory>
#include <iostream>
#include <functional>
#include "omp.h"

#include "utils.hpp"
#include "feature.hpp"
#include "numerics.hpp"
#include "samplers/slice.hpp"
#include "distributions/gamma.hpp"
#include "distributions/inverse_gamma.hpp"
#include "helpers/feature_tree.hpp"
#include "prng.hpp"

namespace baxcat{

class View{
public:
    // Constructors
    // // We don't always need to supply features, we can push them in later
    View(baxcat::PRNG &rng);
    View(std::vector< std::shared_ptr<BaseFeature> > &features, baxcat::PRNG *rng);
    View(std::vector< std::shared_ptr<BaseFeature> > &features, baxcat::PRNG *rng, double alpha,
         std::vector<size_t> row_assignment=std::vector<size_t>(), bool gibbs_init=false);

    // Transitions
    // reassign all rows to categories
    void transitionRows();
    // reassign row
    void transitionRow(size_t row, bool assign_to_max_p_cluster=false);
    // resample CRP parameter
    void transitionCRPAlpha();

    // Probabilities
    // the likelihood of the data in row belonging to the models in cluster
    double rowLogp(size_t row, size_t cluster, bool is_init=false);
    // the likelihood of the data in row belonging to a singleton
    double rowSingletonLogp(size_t row);
    // score of the entire view
    double logScore();

    // adding and removing dims
    // add feature to the view (reassign data, add to lookup)
    void assimilateFeature(std::shared_ptr<BaseFeature> &feature);
    // remove the feature from the view (remove from lookup)
    void releaseFeature(size_t feature_index);

    // setters
    void setRowAssignment(std::vector<size_t> new_row_assignment);
    void appendRow(std::vector<double> data, std::vector<size_t> indices, 
                   bool assign_to_max_p_cluster=false);
    void popRow();

    // getters
    size_t getAssignmentOfRow(size_t row) const;
    size_t getNumFeatures() const;
    size_t getNumRows() const;
    size_t getNumCategories() const;
    double getCRPAlpha() const;
    std::vector<size_t> getRowAssignments() const;
    std::vector<size_t> getClusterCounts() const;
    std::vector<size_t> getFeatureIndices();

    // Debuggind function. Checks that the partitions and the features are not
    // damaged during row transitions
    int checkPartitions();
private:
    // Cleanup
    // destory the singleton cluster, to_destroy, and reassign row to cluster
    // move_to
    void __destroySingletonCluster( size_t row, size_t to_destroy, size_t move_to);
    // move row from cluster current to a new singleton cluster
    void __createSingletonCluster( size_t row, size_t current);
    // move row from cluster move_from to cluster move_to
    void __moveRowToCluster( size_t row, size_t move_from, size_t move_to);
    // init view using gibbs transition
    void __gibbsInit();

    //
    baxcat::PRNG *_rng;
    // number of rows
    size_t _num_rows;
    // the number of clusters/categoiries in the view
    size_t _num_clusters;
    // _num_categories[k] is the number of drows assigned to category k
    std::vector<size_t> _cluster_counts;
    // the CRP parameter for rows to cats
    double _crp_alpha;
    // pointers to feature objects
    baxcat::helpers::FeatureTree _features;
    // _row_assignment[i] is the category [0,K-1] to which row i belongs
    std::vector<size_t> _row_assignment;
    // the score of the view
    double _log_score;
};

} // end namespace baxcat
#endif
