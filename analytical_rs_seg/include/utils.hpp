#ifndef UTILS_HPP
#define UTILS_HPP


#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <utility>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/features/normal_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/common/impl/angles.hpp>


typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

namespace fs = std::filesystem;

namespace utils {

    template<typename T>
    inline typename pcl::PointCloud<T>::Ptr readPointCloud(const std::string& path)
    {
    typename pcl::PointCloud<T>::Ptr cloud(new pcl::PointCloud<T>);
    std::map<std::string, int> ext_map = { {".pcd", 0}, {".ply", 1} };

    switch (ext_map[fs::path(path).extension().string()])
    {
        case 0:
        {
            pcl::PCDReader reader;
            reader.read(path, *cloud);
            break;
        }
        case 1:
        {
            pcl::PLYReader reader;
            reader.read(path, *cloud);
            break;
        }
        default:
        {
            std::cout << "Format not compatible, it should be .pcd or .ply" << std::endl;
            break;
        }
    }

    return cloud;
    }


    struct GtIndices
    {
        pcl::IndicesPtr truss;
        pcl::IndicesPtr ground;

        GtIndices() {
            truss = pcl::IndicesPtr(new std::vector<int>);
            ground = pcl::IndicesPtr(new std::vector<int>);
        }
    };

    struct ConfusionMatrix
    {
        int tp;
        int tn;
        int fp;
        int fn;
    };


    inline pcl::PointCloud<pcl::PointXYZL>::Ptr read_cloud(fs::path _path)
    {

        pcl::PCDReader reader;
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZL>);

        reader.read(_path.string(), *cloud);
        return cloud;
    }

    inline GtIndices get_ground_truth_indices(pcl::PointCloud<pcl::PointXYZL>::Ptr &_cloud)
    {

        GtIndices indices;

        for (size_t i = 0; i < _cloud->points.size(); i++)
        {

            if (_cloud->points[i].label > 0)
            {
                indices.truss->push_back(i);
            }
            else
                indices.ground->push_back(i);
        }
        return indices;
    }


    inline std::vector<int> vector_difference(std::vector<int> v1, std::vector<int> v2)
    {
        std::vector<int> difference;

        // FIRST METHOD - TIME CONSUMING
        // for (int i = 0; i < v1.size(); i++)
        // {
        //     for (int j = 0; j < v2.size(); j++)
        //     {
        //         if (v1[i] != v2[j])
        //         {
        //             difference.push_back(v1[i]);
        //         }
        //     }
        // }

        // SECOND METHOD
        std::unordered_set<int> set_v2(v2.begin(), v2.end());

        for (const int& elem : v1) {
            if (set_v2.find(elem) == set_v2.end()) {
                difference.push_back(elem);
        }
    }

        return difference;
    }

    inline std::vector<int> vector_intersection(std::vector<int> v1, std::vector<int> v2)
    {
        std::vector<int> intersection;

        // FIRST METHOD - TIME CONSUMING
        // for (int i = 0; i < v1.size(); i++)
        // {
        //     for (int j = 0; j < v2.size(); j++)
        //     {
        //         if (v1[i] == v2[j])
        //         {
        //             intersection.push_back(v1[i]);
        //         }
        //     }
        // }

        // SECOND METHOD
        std::unordered_set<int> set_v1(v1.begin(), v1.end());

        for (const int& elem : v2)
        {
            if (set_v1.find(elem) != set_v1.end())
            {
                intersection.push_back(elem);
                set_v1.erase(elem); // Optional: to avoid duplicates in the intersection
            }
        }

        return intersection;
    }


    inline ConfusionMatrix compute_conf_matrix(pcl::IndicesPtr &gt_truss_idx, pcl::IndicesPtr &gt_ground_idx, pcl::IndicesPtr &truss_idx, pcl::IndicesPtr &ground_idx)
    {
        ConfusionMatrix cm;

        cm.tp = vector_intersection(*truss_idx, *gt_truss_idx).size();
        cm.tn = vector_intersection(*ground_idx, *gt_ground_idx).size();
        cm.fp = vector_difference(*truss_idx, *gt_truss_idx).size();
        cm.fn = vector_difference(*ground_idx, *gt_ground_idx).size();

        return cm;
    }


    /**
     * @brief Filtra la nube de puntos en función de los índices pasados como parámetro
     * 
     * @param cloud Nube de entrada
     * @param indices_vec Vector con los índices de los puntos que se quieren extraer
     * @param negative Si es true, se extraen los puntos que no están en indx_vec
     */
    inline PointCloud::Ptr extract_indices(PointCloud::Ptr &cloud, std::vector<pcl::PointIndices> indices_vec, bool negative = false)
    {
    pcl::PointIndices::Ptr indices (new pcl::PointIndices);
    PointCloud::Ptr _cloud_out (new PointCloud);

    for (size_t i = 0; i < indices_vec.size(); i++)
        for(auto index : indices_vec[i].indices)
        indices->indices.push_back(index);
    
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*_cloud_out);

    return _cloud_out;
    }



    /**
     * @brief Filtra la nube de puntos en función de los índices pasados como parámetro
     * 
     * @param cloud Nube de entrada
     * @param indices Indices de los puntos que se quieren extraer
     * @param negative Si es true, se extraen los puntos que no están en indx_vec
     */
    inline PointCloud::Ptr extract_indices (PointCloud::Ptr &_cloud_in, pcl::IndicesPtr &_indices, bool negative = false)
    {
    PointCloud::Ptr _cloud_out (new PointCloud);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(_cloud_in);
    extract.setIndices(_indices);
    extract.setNegative(negative);
    extract.filter(*_cloud_out);

    return _cloud_out;
    }




    /**
     * @brief Returns a Voxelized PointCloud
     * 
     * @param _cloud_in 
     * @return pcl::PointCloud<pcl::PointXYZ>::Ptr
     */
    inline PointCloud::Ptr voxel_filter( PointCloud::Ptr &_cloud_in ,float leafSize = 0.1)
    {
    PointCloud::Ptr _cloud_out (new PointCloud);
    pcl::VoxelGrid<PointT> sor;
    sor.setInputCloud(_cloud_in);
    sor.setLeafSize(leafSize, leafSize, leafSize);
    sor.filter(*_cloud_out);

    return _cloud_out;
    }



    /**
     * @brief Filtra la nube de puntos en función de los índices pasados como parámetro
     * 
     * @param cloud 
     * @param optimizeCoefs 
     * @param distThreshold 
     * @param maxIterations 
     * @return pcl::ModelCoefficients::Ptr 
     */
    inline pcl::ModelCoefficientsPtr compute_planar_ransac (PointCloud::Ptr &_cloud_in, const bool optimizeCoefs,
                float distThreshold = 0.03, int maxIterations = 1000)
    {
    pcl::PointIndices point_indices;
    pcl::SACSegmentation<PointT> ransac;
    pcl::ModelCoefficientsPtr plane_coeffs (new pcl::ModelCoefficients);

    ransac.setInputCloud(_cloud_in);
    ransac.setOptimizeCoefficients(optimizeCoefs);
    ransac.setModelType(pcl::SACMODEL_PLANE);
    ransac.setMethodType(pcl::SAC_RANSAC);
    ransac.setMaxIterations(maxIterations);
    ransac.setDistanceThreshold(distThreshold);
    ransac.segment(point_indices, *plane_coeffs);

    return plane_coeffs;
    }



    /**
     * @brief Filtra la nube de puntos en función de los índices pasados como parámetro
     * 
     * @param cloud 
     * @param coefs 
     * @param distThreshold 
     * @return pcl::PointIndices::Ptr 
     */
    inline std::pair<pcl::IndicesPtr, pcl::IndicesPtr> get_points_near_plane(PointCloud::Ptr &_cloud_in, pcl::ModelCoefficientsPtr &_plane_coeffs, float distThreshold = 0.5f)
    {
    Eigen::Vector4f coefficients(_plane_coeffs->values.data());
    pcl::PointXYZ point;
    pcl::IndicesPtr _plane_inliers (new pcl::Indices);
    pcl::IndicesPtr _plane_outliers (new pcl::Indices);

    for (size_t indx = 0; indx < _cloud_in->points.size(); indx++)
    {
        point = _cloud_in->points[indx];
        float distance = pcl::pointToPlaneDistance(point, coefficients);
        if (pcl::pointToPlaneDistance(point, coefficients) <= distThreshold)
        _plane_inliers->push_back(indx);
        else
        _plane_outliers->push_back(indx);
    }

    return std::pair<pcl::IndicesPtr, pcl::IndicesPtr> {_plane_inliers, _plane_outliers};
    }



    /**
     * @brief Realiza agrupaciones de puntos en función de sus normales
     * 
     * @param cloud  Nube de entrada
     * @return std::vector<pcl::PointIndices> Vector con los indices pertenecientes 
     * a cada agrupación 
     */
    inline std::pair<std::vector<pcl::PointIndices>, int> regrow_segmentation (PointCloud::Ptr &_cloud_in, pcl::IndicesPtr &_indices, YAML::Node config, const bool _visualize=false)
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Estimación de normales
        pcl::PointCloud<pcl::Normal>::Ptr _cloud_normals (new pcl::PointCloud<pcl::Normal>);
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(_cloud_in);
        ne.setInputCloud(_cloud_in);
        // ne.setIndices(_indices);   // Tiene que estar comentado para que la dimension de _cloud_normals sea igual a _cloud_in y funcione regrow
        ne.setSearchMethod(tree);

        if (config["knn"].as<int>() > 0 && config["search_radius"].as<float>() <= 0)
            ne.setKSearch(config["knn"].as<int>());            // Por vecinos no existen normales NaN 30 original
        else if (config["search_radius"].as<float>() > 0 && config["knn"].as<int>() <= 0)
            ne.setRadiusSearch(config["search_radius"].as<float>());  // Por radio existiran puntos cuya normal sea NaN
        else
            ne.setKSearch(30);            // Default value

        ne.compute(*_cloud_normals);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        int num_nan_normals = 0;
        for (const auto& normal : _cloud_normals->points)
        {
            if (!pcl::isFinite(normal))
            {
                num_nan_normals++;
            }
        }

        std::cout << "Number of NaN normals: " << num_nan_normals << " out of " << _cloud_normals->points.size() << std::endl;

        // Segmentación basada en crecimiento de regiones
        std::vector<pcl::PointIndices> _regrow_clusters;
        pcl::RegionGrowing<PointT, pcl::Normal> reg;
        reg.setMinClusterSize (config["min_cluster_size"].as<int>()); //50 original
        reg.setMaxClusterSize (25000);
        reg.setSearchMethod (tree);
        reg.setSmoothModeFlag(false);
        reg.setCurvatureTestFlag(true);
        reg.setResidualThreshold(false);
        reg.setCurvatureThreshold(1); // Lower values lead to more points excluded
        reg.setNumberOfNeighbours (10); //10 original
        reg.setInputCloud (_cloud_in);
        reg.setIndices(_indices);
        reg.setInputNormals (_cloud_normals);
        reg.setSmoothnessThreshold (pcl::deg2rad(config["smoothness"].as<float>())); //10 original
        reg.extract (_regrow_clusters);

        // RESULTS VISUALIZATION 
        if(_visualize)
        {
            pcl::visualization::PCLVisualizer vis ("Clustering Visualizer");
            
            // 2 viewports Normals and clusters
            int v1(0);
            int v2(0);
            vis.createViewPort(0,0,0.5,1, v1);
            vis.createViewPort(0.5,0,1,1, v2);


            pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_cloud (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::ExtractIndices<pcl::PointXYZ> extract1;
            extract1.setInputCloud(_cloud_in);
            pcl::PointIndices::Ptr inliers1 (new pcl::PointIndices);
            inliers1->indices = *_indices;
            extract1.setIndices(inliers1);
            extract1.setNegative(false);
            extract1.filter(*coarse_cloud);

            // Extract normals with the same indices
            pcl::PointCloud<pcl::Normal>::Ptr coarse_normals (new pcl::PointCloud<pcl::Normal>);
            pcl::ExtractIndices<pcl::Normal> extract_normals;
            extract_normals.setInputCloud(_cloud_normals);
            extract_normals.setIndices(inliers1);
            extract_normals.setNegative(false);
            extract_normals.filter(*coarse_normals);

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green_color(coarse_cloud, 10, 150, 10);
            vis.addPointCloud<pcl::PointXYZ>(coarse_cloud, green_color, "cloud", v1);
            vis.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(coarse_cloud, coarse_normals, 5, 0.1, "normals", v1);
            vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 0.0, 0.0, "normals");  // Black normals
            vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, "normals");

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            color_cloud = reg.getColoredCloud();

            pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            extract.setInputCloud(color_cloud);
            pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            inliers->indices = *_indices;
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*color_cloud);

            vis.addPointCloud<pcl::PointXYZRGB>(color_cloud, "Regrow Segments", v2);


            // Single viewport colored clusters
            // pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            // color_cloud = reg.getColoredCloud();

            // pcl::ExtractIndices<pcl::PointXYZRGB> extract;
            // extract.setInputCloud(color_cloud);
            // pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
            // inliers->indices = *_indices;
            // extract.setIndices(inliers);
            // extract.setNegative(false);
            // extract.filter(*color_cloud);

            // vis.addPointCloud<pcl::PointXYZRGB>(color_cloud, "Regrow Segments");

            try
            {
                vis.loadCameraParameters("camera_params_regrow_clusters.txt");
            }
            catch(const std::exception& e)
            {
            }
            
            vis.setBackgroundColor(1,1,1);
            
            while (!vis.wasStopped())
            {
                vis.saveCameraParameters("camera_params_regrow_clusters.txt");
                vis.spinOnce();
            }
        }
        return std::pair<std::vector<pcl::PointIndices>, int> {_regrow_clusters, duration.count()};
    }



    /**
     * @brief Realiza agrupaciones de puntos en función de sus normales
     * 
     * @param cloud  Nube de entrada
     * @return std::std::vector<pcl::PointIndices> std::vector con los indices pertenecientes 
     * a cada agrupación 
     */
    inline std::pair<std::vector<pcl::PointIndices>, int> regrow_segmentation (PointCloud::Ptr &_cloud_in, bool _visualize = false)
    {
    std::cout << "Regrow segmentation complete cloud" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    // Estimación de normales
    pcl::PointCloud<pcl::Normal>::Ptr _cloud_normals (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(_cloud_in);
    ne.setInputCloud(_cloud_in);
    // ne.setIndices(_indices);
    ne.setSearchMethod(tree);
    ne.setKSearch(30);            // Por vecinos no existen normales NaN
    // ne.setRadiusSearch(0.05);  // Por radio existiran puntos cuya normal sea NaN
    ne.compute(*_cloud_normals);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    // Segmentación basada en crecimiento de regiones
    std::vector<pcl::PointIndices> _regrow_clusters;
    pcl::RegionGrowing<PointT, pcl::Normal> reg;
    reg.setMinClusterSize (50); //50 original
    reg.setMaxClusterSize (25000);
    reg.setSearchMethod (tree);
    reg.setSmoothModeFlag(false);
    reg.setCurvatureTestFlag(true);
    reg.setResidualThreshold(false);
    reg.setCurvatureThreshold(1);
    reg.setNumberOfNeighbours (10); //10 original
    reg.setInputCloud (_cloud_in);
    reg.setInputNormals (_cloud_normals);
    reg.setSmoothnessThreshold (pcl::deg2rad(10.0)); //10 original
    reg.extract (_regrow_clusters);

    // std::cout << "Number of clusters: " << _regrow_clusters.size() << std::endl;

    if(_visualize)
    {
        pcl::visualization::PCLVisualizer vis ("Regrow Visualizer");

        int v1(0);
        int v2(0);

        //Define ViewPorts
        vis.createViewPort(0,0,0.5,1, v1);
        vis.createViewPort(0.5,0,1,1, v2);

        pcl::PointCloud<pcl::PointNormal>::Ptr cloud_pn (new pcl::PointCloud<pcl::PointNormal>);
        pcl::concatenateFields (*_cloud_in, *_cloud_normals, *cloud_pn);

        pcl::visualization::PointCloudColorHandler<pcl::PointNormal>::Ptr color_handler;
        color_handler.reset (new pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointNormal> (cloud_pn, "curvature"));
        vis.addPointCloud<pcl::PointNormal>(cloud_pn, *color_handler, "asdf", v1);
        vis.addPointCloudNormals<PointT, pcl::Normal>(_cloud_in, _cloud_normals, 3, 0.1, "normals", v1);


        pcl::PointCloud<pcl::PointXYZRGB>::Ptr color_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        color_cloud = reg.getColoredCloud();
        vis.addPointCloud<pcl::PointXYZRGB>(color_cloud, "Regrow Segments",v2);

        while (!vis.wasStopped())
        vis.spinOnce();
    }

    return std::pair<std::vector<pcl::PointIndices>, int> {_regrow_clusters, duration.count()};
    }




    /**
     * @brief Remove points with less than minNeighbors inside a give radius
     * 
     * @param radius Search sphere radius
     * @param minNeighbors Minimum num of Neighbors to consider a point an inlier
     * @return PointCloud::Ptr Return a PointCloud without low neighbor points
     */
    inline pcl::IndicesPtr radius_outlier_removal (PointCloud::Ptr &_cloud_in, pcl::IndicesPtr &_indices_in, float radius, int minNeighbors, bool negative = false)
    {
    PointCloud::Ptr _cloud_out (new PointCloud);
    pcl::IndicesPtr _indices_out (new pcl::Indices);
    pcl::RadiusOutlierRemoval<PointT> radius_removal;
    radius_removal.setInputCloud(_cloud_in);
    radius_removal.setIndices(_indices_in);
    radius_removal.setRadiusSearch(radius);
    radius_removal.setMinNeighborsInRadius(minNeighbors);
    radius_removal.setNegative(negative);
    radius_removal.filter(*_indices_out);

    return _indices_out;
    }



    inline pcl::IndicesPtr inverseIndices(PointCloud::Ptr &_cloud_in, pcl::IndicesPtr &_indices_in)
    {
    pcl::IndicesPtr _indices_out (new pcl::Indices);
    pcl::ExtractIndices<PointT> extract;
    extract.setInputCloud(_cloud_in);
    extract.setIndices(_indices_in);
    extract.setNegative(true);
    extract.filter(*_indices_out);

    return _indices_out;
    }


    struct eig_decomp
    {
    Eigen::Vector3f values;
    Eigen::Matrix3f vectors;
    };

    inline eig_decomp compute_eigen_decomposition(PointCloud::Ptr &_cloud_in, pcl::IndicesPtr &_indices, bool normalize = true)
    {
    Eigen::Vector4f xyz_centroid;
    PointCloud::Ptr tmp_cloud (new PointCloud);
    tmp_cloud = utils::extract_indices(_cloud_in, _indices);
    pcl::compute3DCentroid(*tmp_cloud, xyz_centroid);

    Eigen::Matrix3f covariance_matrix;
    if (normalize)
        pcl::computeCovarianceMatrixNormalized (*tmp_cloud, xyz_centroid, covariance_matrix); 
    else
        pcl::computeCovarianceMatrix (*tmp_cloud, xyz_centroid, covariance_matrix); 

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);

    eig_decomp eigen_decomp;
    eigen_decomp.values = eigen_solver.eigenvalues();
    eigen_decomp.vectors = eigen_solver.eigenvectors();

    return eigen_decomp;
    }

}
#endif // UTILS_HPP