#pragma once

// C++
#include <iostream>
#include <filesystem>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

// Visualization
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include "arvc_utils/color.hpp"

namespace arvc{

    class Viewer {

    private:
        int element_count;

    public:
        int v1,v2,v3,v4;
        pcl::visualization::PCLVisualizer::Ptr view;
        
        
        // Default constructor
        Viewer();


        // Constructor with name
        Viewer(const std::string& name);
        

        // Destructor
        ~Viewer();


        /**
         * @brief Add a point cloud to the viewer
         * @param cloud Point cloud to be added
         * @param _color Color of the point cloud
         * @param _scale Scale of the point cloud
         * @param viewport Viewport to add the point cloud
        */
        // template <typename T>
        // void addCloud(typename pcl::PointCloud<T>::Ptr& _cloud, const pcl::RGB _color = pcl::RGB(255,255,255), const int& _point_size = 1, const int& _viewport = 0);

        void addCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud, const arvc::Color& _color = arvc::Color(255,255,255), const int& _point_size = 1, const int& _viewport = 0);




        /**
         * @brief Add origin frame to the viewer
         * @param _viewport Viewport to add the origin
         * @param _scale Scale of the origin
         * @param _color Color of the origin
         * @param _viewport Viewport to add the origin
        */
        void addOrigin(const double& _scale = 1.0, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport= 0);

        /**
         * @brief Add a cube to the viewer
         * @param _center_point Center point of the cube
         * @param _size Size of the cube
         * @param _color Color of the cube
         * @param _viewport Viewport to add the cube
        */
        void addCube(const pcl::PointXYZ _center_point, const float& _size, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport=0);


        /**
         * @brief Add a sphere to the viewer
         * @param _center Center of the sphere
         * @param _radius Radius of the sphere
         * @param _color Color of the sphere
         * @param _viewport Viewport to add the sphere
        */
        void addSphere(const pcl::PointXYZ& _center, const float& _radius, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport=0);


        /**
         * @brief Add coordinate system to the viewer
         * @param tf Transformation matrix of the coordinate system
         * @param _scale Scale of the coordinate system
         * @param _viewport Viewport to add the coordinate system
        */
        void addCoordinateSystem(const Eigen::Affine3f& tf, const double _scale = 0.1, const int& _viewport=0);


        /**
         * @brief Set the background color of the viewer
         * @param _color Color of the background
        */
        void setBackgroundColor(const arvc::Color& _color = arvc::Color(255,255,255));


        /**
         * @brief Set the number of viewports
         * @param viewports Number of viewports
        */
        void setViewports(const int& viewports);


        /**
         * @brief Add eigen vectors to the viewer
         * @param _origin Origin of the eigen vectors
         * @param _axis Axis of the eigen vectors
         * @param _viewport Viewport to add the eigen vectors
        */
        void addEigenVectors(const Eigen::Vector3f& _origin, const Eigen::Matrix3f _axis, const int& _viewport=0);


        /**
         * @brief Add a polygon to the viewer
         * @param _polygon Polygon to be added
         * @param _color Color of the polygon
         * @param _viewport Viewport to add the polygon
        */
        void addPolygon(const std::vector<Eigen::Vector3f>& _polygon, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport=0);


        /**
         * @brief Add a plane to the viewer
         * @param _coeffs Coefficients of the plane
         * @param _centroid Centroid of the plane
         * @param _color Color of the plane
         * @param _viewport Viewport to add the plane
        */
        void addPlane(const pcl::ModelCoefficients _coeffs, const Eigen::Vector4f _centroid, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport=0);


        /**
         * @brief Add text to the viewer
         * @param _text Text to be added
         * @param _position Position of the text
         * @param _size Size of the text
         * @param _color Color of the text
         * @param _viewport Viewport to add the text
        */
        void addText(const std::string& _text, const pcl::PointXYZ& _position, const double& _size = 1, const arvc::Color& _color = arvc::Color(255,255,255), const int& _viewport=0);



        void addLine(const pcl::PointXYZ& _p1, const pcl::PointXYZ& _p2, const arvc::Color& _color = BLUE_COLOR, const int& _width =1, const int& _viewport=0);


        /**
         * @brief Close the viewer
        */
        void close();


        /**
         * @brief Remove all elements in the view
        */
        void clear();


        /**
         * @brief Plot the number of elements
        */
        void plot_cloud_count();


        /**
         * @brief Keeps the viewer open until close tab
        */
        void show();

    };
} // namespace arvc