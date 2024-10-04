#include "arvc_utils/viewer.hpp"
// using namespace std;

// namespace arvc{


// Default constructor
arvc::Viewer::Viewer(){
    this->view.reset(new pcl::visualization::PCLVisualizer("ARVC_VIEWER"));
    this->element_count = 0;
    this->v1 = 1;
    this->v2 = 2;
    this->v3 = 3;
    this->v4 = 4;
}


// Constructor with name
arvc::Viewer::Viewer(const std::string& name){
    this->view.reset(new pcl::visualization::PCLVisualizer(name));
    this->element_count = 0;
    this->v1 = 1;
    this->v2 = 2;
    this->v3 = 3;
    this->v4 = 4;

}

// Destructor
arvc::Viewer::~Viewer(){}


/**
 * @brief Add a point cloud to the viewer
 * @param cloud Point cloud to be added
 * @param _color Color of the point cloud
 * @param _scale Scale of the point cloud
 * @param viewport Viewport to add the point cloud
*/
// template <typename T>
// void arvc::Viewer::addCloud(typename pcl::PointCloud<T>::Ptr& _cloud, const pcl::RGB _color, const int& _point_size, const int& _viewport) {
void arvc::Viewer::addCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud, const arvc::Color& _color, const int& _point_size, const int& _viewport) {

    std::string element_name = "cloud_" + std::to_string(this->element_count);

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(_cloud, _color.r, _color.g, _color.b);
    this->view->addPointCloud<pcl::PointXYZ> (_cloud, single_color, element_name, _viewport);
    this->view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, _point_size, element_name, _viewport);

    this->element_count++;
}


/**
 * @brief Add origin frame to the viewer
 * @param _viewport Viewport to add the origin
*/
void arvc::Viewer::addOrigin(const double& _scale, const arvc::Color& _color, const int& _viewport)
{
    std::string element_name = "origin_" + std::to_string(this->element_count);

    this->view->addCoordinateSystem(_scale, element_name, _viewport);

    this->view->addSphere(pcl::PointXYZ(0,0,0), 0.02*_scale, "origin_sphere_"+ std::to_string(this->element_count), _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, _color.r/255, _color.g/255, _color.b/255, "origin_sphere_"+ std::to_string(this->element_count), _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "origin_sphere_"+ std::to_string(this->element_count), _viewport);

    this->view->addText3D("ORIGIN", pcl::PointXYZ(0,0,-0.1*_scale), 0.02*_scale, 0.5, 0.5, 0.5, "origin_text_"+std::to_string(this->element_count), _viewport);

    this->element_count++;
}


/**
 * @brief Add a cube to the viewer
*/
void arvc::Viewer::addCube(const pcl::PointXYZ _center_point, const float& _size, const arvc::Color& _color, const int& _viewport){

    std::string element_name = "cube_" + std::to_string(this->element_count);
    float offset = _size / 2;

    this->view->addCube(_center_point.x - offset, _center_point.x + offset,
                        _center_point.y - offset, _center_point.y + offset,
                        _center_point.z - offset, _center_point.z + offset,
                        _color.r/255, _color.g/255, _color.b/255, element_name, _viewport);

    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, element_name, _viewport);
    this->element_count++;

}



void arvc::Viewer::addSphere(const pcl::PointXYZ& _center, const float& _radius, const arvc::Color& _color, const int& _viewport){
    std::string element_name = "sphere_" + std::to_string(this->element_count);

    this->view->addSphere(_center, _radius, _color.r, _color.g, _color.b, element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, element_name, _viewport);
    this->element_count++;
}


void arvc::Viewer::addCoordinateSystem(const Eigen::Affine3f& tf, const double _scale, const int& _viewport){


    this->view->addCoordinateSystem(0.1, tf, "relative"+std::to_string(this->element_count), _viewport);
    
    pcl::PointXYZ relative_origin(tf.translation().x(), tf.translation().y(), tf.translation().z());

    this->view->addText3D("relative", relative_origin, 0.02*_scale, 1.0, 1.0, 1.0, "relative_text"+std::to_string(this->element_count), _viewport);
    this->element_count++;

}


void arvc::Viewer::setBackgroundColor(const arvc::Color& _color){
    this->view->setBackgroundColor(_color.r/255, _color.g/255, _color.b/255);
}

void arvc::Viewer::setViewports(const int& viewports){

    switch (viewports)
    {
    case 1:
        this->view->createViewPort(0.0, 0.0, 1.0, 1.0, this->v1);
        break;
    case 2:
        this->view->createViewPort(0.0, 0.0, 0.5, 1.0, this->v1);
        this->view->createViewPort(0.5, 0.0, 1.0, 1.0, this->v2);
        break;
    case 3:
        // this->view->createViewPort(0.0, 0.5, 0.5, 1, this->v1);
        // this->view->createViewPort(0.5, 0.5, 1.0, 1, this->v2);
        // this->view->createViewPort(0.0, 0.0, 1.0, 0.5, this->v3);

        this->view->createViewPort(0.0, 0.0, 1/3, 1, this->v1);
        this->view->createViewPort(1/3, 0, 2/3, 1, this->v2);
        this->view->createViewPort(2/3, 0.0, 1.0, 1, this->v3);
    case 4:
        this->view->createViewPort(0.0, 0.5, 0.5, 1, this->v1);
        this->view->createViewPort(0.5, 0.5, 1.0, 1, this->v2);
        this->view->createViewPort(0.0, 0.0, 0.5, 0.5, this->v3);
        this->view->createViewPort(0.5, 0.0, 1.0, 0.5, this->v4);
    default:
        break;
    }
}


void arvc::Viewer::addEigenVectors(const Eigen::Vector3f& _origin, const Eigen::Matrix3f _axis, const int& _viewport){

    pcl::PointXYZ origin(_origin.x(), _origin.y(), _origin.z());
    
    std::vector<pcl::PointXYZ> target_points;

    for (size_t i = 0; i < 3; i++) {
        pcl::PointXYZ target(_axis(0,i) + _origin.x(), _axis(1,i) + _origin.y(), _axis(2,i) + _origin.z());
        target_points.push_back(target);
    }
    

    this->view->addArrow<pcl::PointXYZ, pcl::PointXYZ>(target_points[0], origin, 1.0, 0.0, 0.0, false, "eigenvector_x"+std::to_string(this->element_count), _viewport);
    this->view->addArrow<pcl::PointXYZ, pcl::PointXYZ>(target_points[1], origin, 0.0, 1.0, 0.0, false, "eigenvector_y"+std::to_string(this->element_count), _viewport);
    this->view->addArrow<pcl::PointXYZ, pcl::PointXYZ>(target_points[2], origin, 0.0, 0.0, 1.0, false, "eigenvector_z"+std::to_string(this->element_count), _viewport);
    this->element_count++;

}


void arvc::Viewer::addPolygon(const std::vector<Eigen::Vector3f>& _polygon, const arvc::Color& _color, const int& _viewport){

    std::string element_name = "polygon" + std::to_string(this->element_count);

    // Build the polygon cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr polygon_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    for (auto& point : _polygon) {
        polygon_cloud->push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }

    polygon_cloud->push_back(pcl::PointXYZ(_polygon[0].x(), _polygon[0].y(), _polygon[0].z()));

    // Add the polygon to the viewer
    this->view->addPolygon<pcl::PointXYZ>(polygon_cloud, _color.r/255, _color.g/255, _color.b/255, element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, element_name, _viewport);
    this->element_count++;

}

void arvc::Viewer::addPlane(const pcl::ModelCoefficients _coeffs, const Eigen::Vector4f _centroid, const arvc::Color& _color, const int& _viewport){

    std::string element_name = "plane" + std::to_string(this->element_count);


    // Add the plane to the viewer
    this->view->addPlane(_coeffs, _centroid.x(), _centroid.y(), _centroid.z(), element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, _color.r / 255, _color.g / 255, _color.b / 255, element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, element_name, _viewport);
    
    this->element_count++;
}


void arvc::Viewer::addText(const std::string& _text, const pcl::PointXYZ& _position, const double& _size, const arvc::Color& _color, const int& _viewport){
    std::string element_name = "text" + std::to_string(this->element_count);

    this->view->addText3D(_text, _position, _size, _color.r/255, _color.g/255, _color.b/255, element_name, _viewport);
    this->element_count++;
}


void arvc::Viewer::addLine(const pcl::PointXYZ& _p1, const pcl::PointXYZ& _p2, const arvc::Color& _color, const int& _width, const int& _viewport){
    std::string element_name = "line" + std::to_string(this->element_count);

    this->view->addLine<pcl::PointXYZ>(_p1, _p2, _color.r/255, _color.g/255, _color.b/255, element_name, _viewport);
    this->view->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, _width, element_name, _viewport);
    this->element_count++;
}


void arvc::Viewer::close(){ //NOT WORKING
    this->view->close();
}

void arvc::Viewer::clear(){
    this->view->removeAllPointClouds();
    this->view->removeAllShapes();
    this->element_count = 0;
}

void arvc::Viewer::plot_cloud_count(){
    std::cout << "Cloud Count: " << this->element_count << std::endl;
}

void arvc::Viewer::show(){
    while(!this->view->wasStopped())
    this->view->spinOnce(100);
}


// } // namespace arvc