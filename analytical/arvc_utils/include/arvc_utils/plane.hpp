#pragma once

#include <iostream>
#include <Eigen/Dense>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_normal_plane.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <pcl/features/normal_3d.h>

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgproc.hpp>
// #include "arvc_utils/utils.hpp"

#include "arvc_utils/axes3d.hpp"
#include "arvc_utils/color.hpp"
#include "arvc_utils/console.hpp"

namespace arvc{

class Plane
{
    public:

    pcl::PointIndicesPtr inliers;
    pcl::PointCloud<pcl::PointXYZ>::Ptr original_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    pcl::ModelCoefficientsPtr coeffs;
    
    Eigen::Vector3f normal;
    Eigen::Vector4f centroid;
    Eigen::Affine3f tf;

    
    Eigen::Vector3f eigenvalues;
    arvc::Axes3D eigenvectors;
    
    float length;
    float width;

    std::vector<Eigen::Vector3f> polygon;

    arvc::Color color;
    arvc::Console cons;

    private:
    Eigen::Matrix3f covariance;
    pcl::PointCloud<pcl::PointXYZ>::Ptr projected_cloud;
    
    
    public:

    // Default constructor
    Plane();


    // Copy constructor
    Plane(const Plane& p);


    // Move constructor
    Plane(Plane&& p);


    // Copy assignment
    Plane& operator=(const Plane& p);

    
    // Move assignment
    Plane& operator=(Plane&& p);


    // Destructor
    ~Plane();


    /**
     * @brief Declares a plane object from its coefficients, indices and origin cloud.
     * @param _coeffs Coefficients of the plane.
     * @param _indices Indices of the points that belong to the plane.
     * @param _cloud_in Original cloud.
     * @param _search_directions Axes3D object with the search directions to force the plane directions.
    */
    void setPlane(const pcl::ModelCoefficientsPtr _coeffs, const pcl::PointIndicesPtr& _indices, const pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud_in, const arvc::Axes3D* _search_directions = nullptr);


    /**
     * @brief Returs the cloud of the plane. It extracts the indices from the original cloud that belong to the plane.
     * @return pcl::PointCloud<pcl::PointXYZ>::Ptr Cloud of the plane.
    */
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCloud();


    /**
     * @brief Returns the normal of the plane.
     * @return Eigen::Vector3f Normal of the plane.
    */
    Eigen::Vector3f getNormal();

    /**
     * @brief Forces the eigenvectors to be adjusted to the search directions.
     * @param _search_directions Axes3D object with the search directions to force the plane directions.
    */
    void forceEigenVectors(arvc::Axes3D _search_directions);


    /**
     * @brief Projects the cloud on the plane.
    */
    void projectOnPlane();


    /**
     * @brief Computes the centroid of the plane.
    */
    void getCentroid();


    /**
     * @brief Computes the eigen decomposition of the covariance matrix of the plane.
     * @param force_ortogonal If true, forces the third eigenvector to be perpendicular to the first two.
    */
    void compute_eigenDecomposition(const bool& force_ortogonal = false);


    /**
     * @brief Computes the polygon of the plane.
    */
    void getPolygon();


    /**
     * @brief Computes the transform of the plane.
    */
    void getTransform();


    /**
     * @brief Prints the original eigenvectors.
    */
    void print_original_eigenvectors();


    /**
     * @brief Applies a rigid transformation to all the plane members.
     * @param _tf Transformation to apply.
    */
    void applyTransform(const Eigen::Affine3f& _tf);

    /**
     * @brief Recomputes the coefficients of the plane.
    */
    void recomputeCoeffs();



    /**
     * @brief Applies the plane convention to the plane. 
     * The plane convention is that the normal of the plane is pointing to the origin
     * and the distance to the origin is negative.
    */
    void applyPlaneConvention();



    /**
     * @brief Returns the polygon as a vector of cv::Point3f.
     * @return std::vector<cv::Point3f> Vector of cv::Point3f.
    */
    std::vector<cv::Point3f> getPolygonAsCv();
};

    inline std::ostream& operator<<(std::ostream& os, const Plane& p)
    {

        os << "Parameters:   [ " << p.coeffs->values[0] << ", " << p.coeffs->values[1] << ", " << p.coeffs->values[2] << ", " << p.coeffs->values[3] << " ]" << endl;
        os << "Normal:       [ " << p.normal.x() << ", " << p.normal.y() << ", " << p.normal.z() << " ]" << endl;
        os << "Centroid:     [ " << p.centroid.x() << ", " << p.centroid.y() << ", " << p.centroid.z() << " ]" << endl;
        os << "Indices size: " << p.inliers->indices.size() << endl;
        os << "Cloud size:   " << p.cloud->size() << endl;
        os << "Length: " << p.length << endl;
        os << "Width:  " << p.width << endl;
        os << "Transform:   " << endl << p.tf.matrix() << endl;
        os << "Eigenvalues: \n[ " << p.eigenvalues.x() << ", " << p.eigenvalues.y() << ", " << p.eigenvalues.z() << " ]" << endl;
        os << "Eigenvectors: " << endl << p.eigenvectors;
        os << "Color: " << p.color;
        return os;
    }
}