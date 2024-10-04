#include "arvc_utils/plane.hpp"


namespace arvc{


Plane::Plane(){
    this->coeffs.reset(new pcl::ModelCoefficients);
    this->inliers.reset(new pcl::PointIndices);
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->tf = Eigen::Affine3f::Identity();
    this->eigenvectors = arvc::Axes3D();
    this->eigenvalues = Eigen::Vector3f(0,0,0);

    this->coeffs->values = {0,0,0,0};
    this->inliers->indices = {0};
    this->normal = Eigen::Vector3f(0,0,0);
    this->polygon = std::vector<Eigen::Vector3f>(5);

    this->projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    this->length = 0;
    this->width = 0;
    this->cons.enable = false;
    this->cons.enable_vis = false;

};

// copy constructor
Plane::Plane(const Plane& p)
{
    this->coeffs.reset(new pcl::ModelCoefficients);
    this->inliers.reset(new pcl::PointIndices);
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    *this->coeffs = *p.coeffs;
    *this->inliers = *p.inliers;
    *this->cloud = *p.cloud;
    *this->original_cloud = *p.original_cloud;
    *this->projected_cloud = *p.projected_cloud;

    this->normal = p.normal;
    this->centroid = p.centroid;
    this->tf = p.tf;
    this->eigenvectors = p.eigenvectors;
    this->eigenvalues = p.eigenvalues;
    this->polygon = p.polygon;
    this->length = p.length;
    this->width = p.width;
    this->color = p.color;
};


Plane::Plane(Plane&& p)
{
    this->coeffs.reset(new pcl::ModelCoefficients);
    this->inliers.reset(new pcl::PointIndices);
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    *this->coeffs = *p.coeffs;
    *this->inliers = *p.inliers;
    *this->cloud = *p.cloud;
    *this->original_cloud = *p.original_cloud;
    *this->projected_cloud = *p.projected_cloud;

    this->normal = p.normal;
    this->centroid = p.centroid;
    this->tf = p.tf;
    this->eigenvectors = p.eigenvectors;
    this->eigenvalues = p.eigenvalues;
    this->polygon = p.polygon;
    this->length = p.length;
    this->width = p.width;
    this->color = p.color;
};


// copy assignment
Plane& Plane::operator=(const Plane& p)
{
    this->coeffs.reset(new pcl::ModelCoefficients);
    this->inliers.reset(new pcl::PointIndices);
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    *this->coeffs = *p.coeffs;
    *this->inliers = *p.inliers;
    *this->cloud = *p.cloud;
    *this->original_cloud = *p.original_cloud;
    *this->projected_cloud = *p.projected_cloud;

    this->normal = p.normal;
    this->centroid = p.centroid;
    this->tf = p.tf;
    this->eigenvectors = p.eigenvectors;
    this->eigenvalues = p.eigenvalues;
    this->polygon = p.polygon;
    this->length = p.length;
    this->width = p.width;
    this->color = p.color;

    return *this;
};

// move assignment
Plane& Plane::operator=(Plane&& p)
{
    this->coeffs.reset(new pcl::ModelCoefficients);
    this->inliers.reset(new pcl::PointIndices);
    this->cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->original_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    this->projected_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    *this->coeffs = *p.coeffs;
    *this->inliers = *p.inliers;
    *this->cloud = *p.cloud;
    *this->original_cloud = *p.original_cloud;
    *this->projected_cloud = *p.projected_cloud;

    this->normal = p.normal;
    this->centroid = p.centroid;
    this->tf = p.tf;
    this->eigenvectors = p.eigenvectors;
    this->eigenvalues = p.eigenvalues;
    this->polygon = p.polygon;
    this->length = p.length;
    this->width = p.width;
    this->color = p.color;

    return *this;
};


Plane::~Plane(){
    this->coeffs->values = {0,0,0,0};
    this->inliers->indices = {0};
};


void Plane::setPlane(const pcl::ModelCoefficientsPtr _coeffs, const pcl::PointIndicesPtr& _indices, const pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud_in, const arvc::Axes3D* _search_directions) {
    if (_search_directions == nullptr) {
        this->cons.debug("Setting plane with normal mode.");
    } else {
        this->cons.debug("Setting plane with forced mode.");
    }

    *this->coeffs = *_coeffs;
    *this->inliers = *_indices;
    *this->original_cloud = *_cloud_in;
    this->getNormal();
    this->getCloud();
    this->getCentroid();
    this->compute_eigenDecomposition(true);

    if (_search_directions != nullptr) {
        this->forceEigenVectors(*_search_directions);
    }

    this->getTransform();  
    this->getPolygon();
    this->color.random();
}



pcl::PointCloud<pcl::PointXYZ>::Ptr Plane::getCloud(){
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    
    extract.setInputCloud(this->original_cloud);
    extract.setIndices(this->inliers);
    extract.setNegative(false);
    extract.filter(*this->cloud);

    return this->cloud;
};


/**
 * @brief Returns the normal of the plane.
 * @return Eigen::Vector3f Normal of the plane.
*/
Eigen::Vector3f Plane::getNormal(){
    this->normal = Eigen::Vector3f(this->coeffs->values[0], this->coeffs->values[1], this->coeffs->values[2]);
    return normal;
}


/**
 * @brief Forces the eigenvectors to be adjusted to the search directions.
 * @param _search_directions Axes3D object with the search directions to force the plane directions.
*/
void Plane::forceEigenVectors(arvc::Axes3D _search_directions){
    this->cons.debug("Forcing eigenvectors to be adjusted to the search directions");
    
    int idx;
    std::vector<float>::iterator it;
    std::vector<float> dp; // dot product
    arvc::Axes3D _new_axes;


    // FOR EACH EIGENVECTOR, FIND THE MOST SIMILAR SEARCH DIRECTION
    for (size_t i = 0; i < 3; i++)
    {
        dp.clear(); // clear/reset the dot product vector

        // FOR EACH SEARCH DIRECTION, COMPUTE THE DOT PRODUCT WITH THE EIGENVECTOR
        for (size_t j = 0; j < 3; j++){
            dp.push_back(this->eigenvectors(i).dot(_search_directions(j)));
            this->cons.debug("Eigenvector_" + std::to_string(i) + " · Direction_" + std::to_string(j) + " = " + std::to_string(dp[j]));
        }

        // THE MAXIMUM DOT PRODUCT, IN ABSOLUTE VALUE IS THE MOST SIMILAR SEARCH DIRECTION
        it = std::max_element(dp.begin(), dp.end(), [](const auto& a, const auto& b) {
            return std::abs(a) < std::abs(b);
        });

        idx = distance(dp.begin(), it);
        _new_axes(i) = _search_directions(idx);
    }

    this->eigenvectors.x = _new_axes.x;
    this->eigenvectors.y = _new_axes.y;
    this->eigenvectors.z = _new_axes.z;
}


/**
 * @brief Projects the cloud on the plane.
*/
void Plane::projectOnPlane(){

    pcl::ProjectInliers<pcl::PointXYZ> proj;
    proj.setModelType(pcl::SACMODEL_PLANE);
    proj.setInputCloud(this->cloud);
    proj.setModelCoefficients(this->coeffs);
    proj.filter(*this->projected_cloud);
}


/**
 * @brief Computes the centroid of the plane.
*/
void Plane::getCentroid(){
    pcl::compute3DCentroid(*this->cloud, this->centroid);
}


/**
 * @brief Computes the eigen decomposition of the covariance matrix of the plane.
 * @param force_ortogonal If true, forces the third eigenvector to be perpendicular to the first two.
*/
void Plane::compute_eigenDecomposition(const bool& force_ortogonal){
    arvc::Axes3D _axes;
    pcl::compute3DCentroid(*this->cloud, this->centroid);
    pcl::computeCovarianceMatrixNormalized(*this->cloud, this->centroid, this->covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(this->covariance, Eigen::ComputeEigenvectors);

    // EIGEN VECTORS
    Eigen::Matrix3f eigDx = eigen_solver.eigenvectors();
    
    // Forzar a que el tercer vector sea perpendicular a los anteriores.
    if (force_ortogonal)
        eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));
    

    this->eigenvectors.x = eigDx.col(2);
    this->eigenvectors.y = eigDx.col(1);
    this->eigenvectors.z = eigDx.col(0);

    // EIGEN VALUES
    this->eigenvalues = eigen_solver.eigenvalues().reverse();
}


/**
 * @brief Computes the polygon of the plane.
*/
void Plane::getPolygon(){
    pcl::PointCloud<pcl::PointXYZ>::Ptr relative_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    this->projectOnPlane();
    pcl::transformPointCloud(*this->projected_cloud, *relative_cloud, this->tf.inverse());

    if (this->cons.enable){
        cout << "Relative cloud size: " << relative_cloud->size() << endl;
        cout << "Projected cloud size: " << this->projected_cloud->size() << endl;
    }

    pcl::PointXYZ max_point;
    pcl::PointXYZ min_point;

    pcl::getMinMax3D(*relative_cloud, min_point, max_point);

    if (this->cons.enable){
        cout << "Min point: " << min_point << endl;
        cout << "Max point: " << max_point << endl;

    }

    this->polygon[0] = Eigen::Vector3f(min_point.x, min_point.y, 0.0);
    this->polygon[1] = Eigen::Vector3f(min_point.x, max_point.y, 0.0);
    this->polygon[2] = Eigen::Vector3f(max_point.x, max_point.y, 0.0);
    this->polygon[3] = Eigen::Vector3f(max_point.x, min_point.y, 0.0);
    this->polygon[4] = this->polygon[0];

    this->length = abs(this->polygon[1].x() - this->polygon[2].x()); 
    this->width = abs(this->polygon[0].y() - this->polygon[1].y());

    if (this->length < this->width){
        std::swap(this->length, this->width);
        std::swap(this->eigenvectors.x, this->eigenvectors.y);
    }


    for (int i = 0; i < this->polygon.size(); i++)
        this->polygon[i] = this->tf * this->polygon[i];
}


/**
 * @brief Computes the transform of the plane.
*/
void Plane::getTransform(){
    this->tf.translation() = this->centroid.head<3>();
    this->tf.linear() = this->eigenvectors.getRotationMatrix();
}




void Plane::applyTransform(const Eigen::Affine3f& _tf){
    pcl::transformPointCloud(*this->original_cloud, *this->original_cloud, _tf);
    this->getCloud();
    this->getCentroid();
    this->eigenvectors.getMatrix3f() = _tf.linear() * this->eigenvectors.getMatrix3f();
    this->compute_eigenDecomposition(true);
    this->recomputeCoeffs();
    this->getNormal();
    this->tf = _tf * this->tf;
    this->getPolygon();
}

void Plane::recomputeCoeffs() {

    pcl::PointIndicesPtr dummy(new pcl::PointIndices);

    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.02);
    seg.setInputCloud(this->cloud);
    seg.segment(*dummy, *this->coeffs);
    Eigen::Vector4f _coeffs{this->coeffs->values[0], this->coeffs->values[1], this->coeffs->values[2], this->coeffs->values[3]};
    pcl::PointXYZ _centroid{this->centroid.x(), this->centroid.y(), this->centroid.z()};

    pcl::flipNormalTowardsViewpoint(_centroid, 0, 0, 0, _coeffs);
    this->coeffs->values = {_coeffs[0], _coeffs[1], _coeffs[2], _coeffs[3]};
    // this->applyPlaneConvention();
}


std::vector<cv::Point3f> Plane::getPolygonAsCv(){

    std::vector<cv::Point3f> cv_points;
    for (size_t i = 0; i < this->polygon.size(); i++)
        cv_points.push_back(cv::Point3f(this->polygon[i].x(), this->polygon[i].y(), this->polygon[i].z()));

    return cv_points;
}


void Plane::applyPlaneConvention(){
    if (this->coeffs->values[3] < 0){
        this->coeffs->values[0] *= -1;
        this->coeffs->values[1] *= -1;
        this->coeffs->values[2] *= -1;
        this->coeffs->values[3] *= -1;
    }
}


} // namespace arvc