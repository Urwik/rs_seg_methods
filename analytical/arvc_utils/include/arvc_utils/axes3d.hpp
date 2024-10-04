#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <pcl/point_types.h>


namespace arvc{
/**
 * @brief Class representing a 3D coordinate system with three orthogonal axes.
 */

class Axes3D
{
public:
  /************************************************************************
   * VARIABLES ************************************************************
   ************************************************************************/
  Eigen::Matrix3f rot_matrix; ///< The rotation matrix of the axes.
  Eigen::Vector3f x;          ///< The x-axis.
  Eigen::Vector3f y;          ///< The y-axis.
  Eigen::Vector3f z;          ///< The z-axis.


  /************************************************************************
   * FUNCTIONS ************************************************************
   ************************************************************************/
  /**
   * @brief Default constructor. Initializes the axes with the standard basis vectors and identity rotation matrix.
   */
  Axes3D();


  /**
   * @brief Destructor.
   */
  ~Axes3D();


  /**
   * @brief Accessor function to retrieve the axes by index.
   * @param index The index of the axis (0 for x, 1 for y, 2 for z).
   * @return A reference to the requested axis.
   * @throws std::out_of_range if the index is out of range.
   */
  Eigen::Vector3f& operator()(int index);


  /**
   * @brief Set the x-axis.
   * @param _x The new x-axis.
  */
  void setX(Eigen::Vector3f _x);


  /**
   * @brief Set the y-axis.
   * @param _y The new y-axis.
  */
  void setY(Eigen::Vector3f _y);


  /**
   * @brief Set the z-axis.
   * @param _z The new z-axis.
  */
  void setZ(Eigen::Vector3f _z);


  /**
   * @brief Computes a 3D point by adding a given vector to a centroid.
   * @param _vector The vector to be added.
   * @param _centroid The centroid to which the vector is added.
   * @return The computed 3D point.
   */
  pcl::PointXYZ getPoint(Eigen::Vector3f _vector, Eigen::Vector4f _centroid);


  /**
   * @brief Computes and returns the rotation matrix of the axes.
   * @return The rotation matrix.
   */
  Eigen::Matrix3f getRotationMatrix();


  /**
   * @brief Returns the axes as a 3x3 matrix.
  */
  Eigen::Matrix3f getMatrix3f();
};

/**
 * @brief Overload of the << operator to output the values of the axes.
 * @param os The output stream.
 * @param a The axes object.
 * @return The output stream.
 */
inline std::ostream& operator<<(std::ostream &os, const Axes3D &a) {
    // set precision for floating point output
    os << std::fixed;
    os.precision(2);

    os << "X: [ " << a.x.x() << ", " << a.x.y() << ", " << a.x.z() << " ]" << std::endl;
    os << "Y: [ " << a.y.x() << ", " << a.y.y() << ", " << a.y.z() << " ]" << std::endl;
    os << "Z: [ " << a.z.x() << ", " << a.z.y() << ", " << a.z.z() << " ]" << std::endl;

    return os;
}

} // namespace arvc