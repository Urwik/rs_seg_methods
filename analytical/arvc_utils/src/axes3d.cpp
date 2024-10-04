#include "arvc_utils/axes3d.hpp"


/**
 * @brief Default constructor. Initializes the axes with the standard basis vectors and identity rotation matrix.
 */
arvc::Axes3D::Axes3D() {
    this->x = Eigen::Vector3f(1, 0, 0);
    this->y = Eigen::Vector3f(0, 1, 0);
    this->z = Eigen::Vector3f(0, 0, 1);
    this->rot_matrix = Eigen::Matrix3f::Identity();
}



/**
 * @brief Destructor.
 */
arvc::Axes3D::~Axes3D() {
    this->x = Eigen::Vector3f(0, 0, 0);
    this->y = Eigen::Vector3f(0, 0, 0);
    this->z = Eigen::Vector3f(0, 0, 0);
    this->rot_matrix = Eigen::Matrix3f::Zero();
}



/**
 * @brief Accessor function to retrieve the axes by index.
 * @param index The index of the axis (0 for x, 1 for y, 2 for z).
 * @return A reference to the requested axis.
 * @throws std::out_of_range if the index is out of range.
 */
Eigen::Vector3f& arvc::Axes3D::operator()(int index) {

    switch (index)
    {
        case 0:
            return this->x;
        case 1:
            return this->y;
        case 2:
            return this->z;
        default:
            throw std::out_of_range("Index out of range");
    }
}


/**
 * @brief Sets the x-axis.
 * @param _x The new x-axis as an Eigen::Vector3f.
*/
void arvc::Axes3D::setX(Eigen::Vector3f _x) {
    this->x = _x;
}


/**
 * @brief Sets the y-axis.
 * @param _y The new y-axis as an Eigen::Vector3f.
*/
void arvc::Axes3D::setY(Eigen::Vector3f _y) {
    this->y = _y;
}


/**
 * @brief Sets the z-axis.
 * @param _z The new z-axis as an Eigen::Vector3f.
*/
void arvc::Axes3D::setZ(Eigen::Vector3f _z) {
    this->z = _z;
}



/**
 * @brief Computes a 3D point by adding a given vector to a centroid.
 * @param _vector The vector to be added.
 * @param _centroid The centroid to which the vector is added.
 * @return The computed 3D point.
 */
pcl::PointXYZ arvc::Axes3D::getPoint(Eigen::Vector3f _vector, Eigen::Vector4f _centroid) {
    
    pcl::PointXYZ point;
    point.x = _vector(0) + _centroid(0);
    point.y = _vector(1) + _centroid(1);
    point.z = _vector(2) + _centroid(2);

    return point;
}

/**
 * @brief Computes and returns the rotation matrix of the axes.
 * @return The rotation matrix.
 */
Eigen::Matrix3f arvc::Axes3D::getRotationMatrix() {

    this->rot_matrix << x.x(), y.x(), z.x(),
        x.y(), y.y(), z.y(),
        x.z(), y.z(), z.z();

    return this->rot_matrix;
}


Eigen::Matrix3f arvc::Axes3D::getMatrix3f(){
    
    Eigen::Matrix3f matrix;
    matrix.col(0) = this->x;
    matrix.col(1) = this->y;
    matrix.col(2) = this->z;

    return matrix;
}

