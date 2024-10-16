#ifndef GROUND_FILTER_HPP
#define GROUND_FILTER_HPP

#include <iostream>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <yaml-cpp/yaml.h>

#include "tqdm.hpp"
#include "utils.hpp"
#include "metrics.hpp"
#include "console.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>

namespace fs = std::filesystem;

using namespace std;

enum class MODE{
    RATIO,
    MAGNITUDE,
    HYBRID,
    WOFINE,
    WOCOARSE_RATIO,
    WOCOARSE_MAGNITUDE,
    WOCOARSE_HYBRID
};

class GroundFilter
{

public:
    MODE mode;
    
    pcl::IndicesPtr gt_truss_idx;
    pcl::IndicesPtr gt_ground_idx;

    pcl::IndicesPtr coarse_ground_idx;
    pcl::IndicesPtr coarse_truss_idx;

    pcl::IndicesPtr truss_idx;
    pcl::IndicesPtr ground_idx;
    pcl::IndicesPtr low_density_idx;
    pcl::IndicesPtr wrong_idx;

    YAML::Node cfg;
    string cloud_id;
    fs::path save_cloud_path;

    vector<pcl::PointIndices> regrow_clusters;
    vector<pcl::PointIndices> euclid_clusters;
    vector<int> valid_clusters;


    // TODO: REMOVE ARVC LIBRARY USAGE
    utils::Console cons;

    utils::Metrics metrics;
    utils::ConfusionMatrix cm;

    int normals_time;
    int metrics_time;

    bool enable_metrics;
    bool enable_density_filter;
    bool enable_euclidean_clustering;

    float ratio_threshold;
    float magnitude_threshold;
    float ransac_threshold;
    float voxel_size;
    bool density_first;
    float density_radius;
    int density_threshold;

    float cluster_radius;
    int cluster_min_size;
    bool save_cloud;


private:
    PointCloud::Ptr cloud_in;
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_in_labeled;
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_out;


    pcl::IndicesPtr tp_idx, fp_idx, fn_idx, tn_idx;

    float node_length, node_width, sac_threshold;

public:

    GroundFilter();

    GroundFilter(YAML::Node _cfg);

    ~GroundFilter();

    void set_input_cloud(pcl::PointCloud<pcl::PointXYZL>::Ptr &_cloud);

    void set_mode(MODE _mode);

    void set_node_length(float _length);

    void set_node_width(float _width);

    void set_sac_threshold(float _threshold);

    void set_voxel_size(float _voxel_size);

    int compute();

    void save_cloud_result();

    float get_ground_data_ratio();

private:

    void coarse_segmentation();

    void fine_segmentation();

    void density_filter();

    void euclidean_clustering();

    bool valid_ratio(pcl::IndicesPtr &_cluster_indices);

    bool valid_magnitude(pcl::IndicesPtr &_cluster_indices);

    vector<int> validate_clusters_by_ratio();

    vector<int> validate_clusters_by_magnitude();

    vector<int> validate_clusters_hybrid();

    void validate_clusters();

    void update_segmentation();

    /**
     * @brief Computes the confusion matrix using the ground truth and the computed segmentation
     * @return Returns the TP, FP, FN, TN indexes
    */
    void getConfMatrixIndexes();

    void compute_metrics();

    void view_final_segmentation();
};


inline MODE parse_MODE(const std::string& mode) {
    if (mode == "wofine") {
        return MODE::WOFINE;
    } else if (mode == "wocoarse_ratio") {
        return MODE::WOCOARSE_RATIO;
    } else if (mode == "wocoarse_magnitude") {
        return MODE::WOCOARSE_MAGNITUDE;
    } else if (mode == "wocoarse_hybrid") {
        return MODE::WOCOARSE_HYBRID;
    } else if (mode == "ratio") {
        return MODE::RATIO;
    } else if (mode == "magnitude") {
        return MODE::MAGNITUDE;
    } else if (mode == "hybrid") {
        return MODE::HYBRID;
    } else {
        return MODE::HYBRID;
    }
}

inline string parse_MODE(MODE mode) {
    switch (mode) {
        case MODE::WOFINE:
            return "wofine";
        case MODE::WOCOARSE_RATIO:
            return "wocoarse_ratio";
        case MODE::WOCOARSE_MAGNITUDE:
            return "wocoarse_magnitude";
        case MODE::WOCOARSE_HYBRID:
            return "wocoarse_hybrid";
        case MODE::RATIO:
            return "ratio";
        case MODE::MAGNITUDE:
            return "magnitude";
        case MODE::HYBRID:
            return "hybrid";
        default:
            return "none";
    }
}

#endif // GROUND_FILTER_HPP