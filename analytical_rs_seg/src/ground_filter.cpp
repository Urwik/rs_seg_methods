#include "analytical_rs_seg/ground_filter.hpp"

using namespace std;

// Private methods
void GroundFilter::coarse_segmentation()
{
	this->cons.debug("Coarse segmentation", "GREEN");
	PointCloud::Ptr tmp_cloud(new PointCloud);
	pcl::ModelCoefficientsPtr tmp_plane_coefss(new pcl::ModelCoefficients);

	tmp_cloud = utils::voxel_filter(this->cloud_in, this->voxel_size);
	tmp_plane_coefss = utils::compute_planar_ransac(tmp_cloud, true, this->sac_threshold, 1000);
	auto coarse_indices = utils::get_points_near_plane(this->cloud_in, tmp_plane_coefss, this->sac_threshold);
	this->coarse_ground_idx = coarse_indices.first;
	this->coarse_truss_idx = coarse_indices.second;

	if (this->cfg["COARSE"]["visualize"].as<bool>() && this->cons.enable_vis)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_ground(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_truss(new pcl::PointCloud<pcl::PointXYZ>);

		pcl::PointCloud<pcl::PointXYZ>::Ptr wrong_truss_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		ConfusionMatrixIndexes cm_indexes = this->getConfMatrixIndexes(this->coarse_truss_idx, this->coarse_ground_idx);

		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(this->cloud_in);
		extract.setIndices(this->coarse_ground_idx);
		extract.setNegative(false);
		extract.filter(*coarse_ground);

		extract.setIndices(this->coarse_truss_idx);
		extract.filter(*coarse_truss);

		extract.setIndices(cm_indexes.fp);
		extract.filter(*wrong_truss_cloud);

		this->cons.debug("Visualizing coarse segmentation", "GREEN");
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("COARSE SEGMENTATION - " + this->cloud_id));


		// Dual Viewports
		// int v1(0);
		// int v2(1);
		// viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
		// viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
		// viewer->setBackgroundColor(1, 1, 1);

		// viewer->addPointCloud<pcl::PointXYZ>(coarse_ground, ground_color, "coarse_ground", v1);
		// viewer->addPointCloud<pcl::PointXYZ>(coarse_truss, truss_color, "coarse_truss", v1);

		// viewer->addPointCloud<pcl::PointXYZ>(coarse_truss, truss_color, "coarse_truss", v2);
		// viewer->addPointCloud<pcl::PointXYZ>(wrong_truss_cloud, wrong_color, "wrong_truss", v2);


		// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "coarse_ground", v1);
		// viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "coarse_truss", v1);

		// Single Viewport
		viewer->setBackgroundColor(1, 1, 1);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> ground_color(coarse_ground, 100, 100, 100);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> truss_color(coarse_truss, 50, 190, 50);
		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> wrong_color(wrong_truss_cloud, 200, 10, 10);

		// viewer->addPointCloud<pcl::PointXYZ>(this->cloud_in, "cloud");

		viewer->addPointCloud<pcl::PointXYZ>(coarse_ground, ground_color, "coarse_ground");
		viewer->addPointCloud<pcl::PointXYZ>(coarse_truss, truss_color, "coarse_truss");

		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "coarse_ground");
		viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "coarse_truss");

		while (!viewer->wasStopped())
		{
			viewer->spinOnce();
		}
	}
}

void GroundFilter::fine_segmentation()
{
	this->cons.debug("Fine segmentation", "GREEN");

	std::pair<vector<pcl::PointIndices>, int> regrow_output;

	if (this->coarse_ground_idx->size() > 0)
		regrow_output = utils::regrow_segmentation(this->cloud_in, this->coarse_ground_idx, this->cfg["REGROW"], this->cfg["REGROW"]["visualize"].as<bool>());
	else
		regrow_output = utils::regrow_segmentation(this->cloud_in, this->cfg["REGROW"]["visualize"].as<bool>());

	this->regrow_clusters = regrow_output.first;
	this->normals_time = regrow_output.second;

	this->validate_clusters();

	// Append valid clusters to truss indices
	for (int clus_indx : this->valid_clusters)
		this->coarse_truss_idx->insert(this->coarse_truss_idx->end(), this->regrow_clusters[clus_indx].indices.begin(), this->regrow_clusters[clus_indx].indices.end());


	if (this->cfg["FINE"]["visualize"].as<bool>() && this->cons.enable_vis)
	{
		this->cons.debug("Visualizing fine segmentation", "GREEN");
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("FINE SEGMENTATION - " + this->cloud_id));

		ConfusionMatrixIndexes cm_indexes = this->getConfMatrixIndexes(this->coarse_truss_idx, this->coarse_ground_idx);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr fp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr fn_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		viewer->setBackgroundColor(1, 1, 1);

		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(this->cloud_in);
		extract.setNegative(false);

		extract.setIndices(cm_indexes.tp);
		extract.filter(*tp_cloud);
		extract.setIndices(cm_indexes.tn);
		extract.filter(*tn_cloud);
		extract.setIndices(cm_indexes.fp);
		extract.filter(*fp_cloud);
		extract.setIndices(cm_indexes.fn);
		extract.filter(*fn_cloud);

		viewer->addCoordinateSystem(1.5);

		viewer->addPointCloud<pcl::PointXYZ>(tp_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(tp_cloud, 50, 190, 50), "TP Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(tn_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(tn_cloud, 100, 100, 100), "TN Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(fp_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(fp_cloud, 200, 10, 10), "FP Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(fn_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(fn_cloud, 100, 100, 100), "FN Cloud");

		while (!viewer->wasStopped())
		{
			viewer->spinOnce();
		}
	}
	
}

void GroundFilter::validate_clusters()
{
	this->cons.debug("Validating clusters");

	map<string, int> mode_dict;
	mode_dict["ratio"] = 0;
	mode_dict["magnitude"] = 1;
	mode_dict["hybrid"] = 2;

	switch (this->mode)
	{
	case MODE::RATIO:
		this->valid_clusters = this->validate_clusters_by_ratio();
		break;
	case MODE::MAGNITUDE:
		this->valid_clusters = this->validate_clusters_by_magnitude();
		break;
	case MODE::HYBRID:
		this->valid_clusters = this->validate_clusters_hybrid();
		break;
	case MODE::WOCOARSE_RATIO:
		this->valid_clusters = this->validate_clusters_by_ratio();
		break;
	case MODE::WOCOARSE_MAGNITUDE:
		this->valid_clusters = this->validate_clusters_by_magnitude();
		break;
	case MODE::WOCOARSE_HYBRID:
		this->valid_clusters = this->validate_clusters_hybrid();
		break;
	case MODE::WOFINE:
		break;
	default:
		break;
	}
}

void GroundFilter::density_filter()
{

	// if (this->density_first){
	//   this->cons.debug("Density filter FIRST");
	//   pcl::RadiusOutlierRemoval<pcl::PointXYZL> radius_removal;
	//   radius_removal.setInputCloud(this->cloud_in_labeled);
	//   radius_removal.setRadiusSearch(this->density_radius);
	//   radius_removal.setMinNeighborsInRadius(this->density_threshold);
	//   radius_removal.setNegative(false);
	//   radius_removal.filter(*this->cloud_in_labeled);
	// }
	// else {
	this->cons.debug("Applying Density Filter");
	this->truss_idx = utils::radius_outlier_removal(this->cloud_in, this->truss_idx, this->density_radius, this->density_threshold, false);
	this->ground_idx = utils::inverseIndices(this->cloud_in, this->truss_idx);
	// }

	if (this->cfg["DENSITY"]["visualize"].as<bool>())
	{
		this->cons.debug("Visualizing density segmentation", "GREEN");
		pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("DENSITY SEGMENTATION - " + this->cloud_id));

		ConfusionMatrixIndexes cm_indexes = this->getConfMatrixIndexes(this->truss_idx, this->ground_idx);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr tn_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr fp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr fn_cloud(new pcl::PointCloud<pcl::PointXYZ>);

		viewer->setBackgroundColor(1, 1, 1);

		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud(this->cloud_in);
		extract.setNegative(false);

		extract.setIndices(cm_indexes.tp);
		extract.filter(*tp_cloud);
		extract.setIndices(cm_indexes.tn);
		extract.filter(*tn_cloud);
		extract.setIndices(cm_indexes.fp);
		extract.filter(*fp_cloud);
		extract.setIndices(cm_indexes.fn);
		extract.filter(*fn_cloud);

		viewer->addCoordinateSystem(1.5);

		viewer->addPointCloud<pcl::PointXYZ>(tp_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(tp_cloud, 50, 190, 50), "TP Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(tn_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(tn_cloud, 100, 100, 100), "TN Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(fp_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(fp_cloud, 200, 10, 10), "FP Cloud");
		viewer->addPointCloud<pcl::PointXYZ>(fn_cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(fn_cloud, 100, 100, 100), "FN Cloud");

		while (!viewer->wasStopped())
		{
			viewer->spinOnce();
		}
	}
}

void GroundFilter::euclidean_clustering()
{
	this->cons.debug("Euclidean clustering");

	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(this->cloud_in);

	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance(0.5);
	ec.setMinClusterSize(100);
	ec.setMaxClusterSize(25000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(this->cloud_in);
	ec.setIndices(this->truss_idx);
	ec.extract(this->euclid_clusters);

	// Get largest cluster
	int largest_cluster_size = 0;
	int largest_cluster_idx = 0;
	for (int i = 0; i < this->euclid_clusters.size(); i++)
	{
		if (this->euclid_clusters[i].indices.size() > largest_cluster_size)
		{
			largest_cluster_size = this->euclid_clusters[i].indices.size();
			largest_cluster_idx = i;
		}
	}

	*this->truss_idx = this->euclid_clusters[largest_cluster_idx].indices;
	*this->ground_idx = *utils::inverseIndices(this->cloud_in, this->truss_idx);
}

void GroundFilter::update_segmentation()
{
	// Update truss and ground indices for final segmentation
	*this->truss_idx = *this->coarse_truss_idx;
	*this->ground_idx = *utils::inverseIndices(this->cloud_in, this->truss_idx);
}

bool GroundFilter::valid_ratio(pcl::IndicesPtr &_cluster_indices)
{
	auto eig_decomp = utils::compute_eigen_decomposition(this->cloud_in, _cluster_indices, true);
	float size_ratio = eig_decomp.values(1) / eig_decomp.values(2);

	if (size_ratio <= this->ratio_threshold)
	{
		return true;
	}
	else
		return false;
}

bool GroundFilter::valid_magnitude(pcl::IndicesPtr &_cluster_indices)
{

	// GET THE CLOUD REPRESENTING THE CLUSTER
	PointCloud::Ptr cluster_cloud(new PointCloud);
	cluster_cloud = utils::extract_indices(this->cloud_in, _cluster_indices);

	// COMPUTE CLOUD CENTROID
	Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cluster_cloud, centroid);

	// COMPUTE EIGEN DECOMPOSITION
	utils::eig_decomp eigen_decomp = utils::compute_eigen_decomposition(this->cloud_in, _cluster_indices, false);

	// Compute transform between the original cloud and the eigen vectors of the cluster
	Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
	projectionTransform.block<3, 3>(0, 0) = eigen_decomp.vectors.transpose();
	projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * centroid.head<3>());

	// Transform the origin to the centroid of the cluster and to its eigen vectors
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cluster_cloud, *cloudPointsProjected, projectionTransform);

	// Get the minimum and maximum points of the transformed cloud.
	pcl::PointXYZ minPoint, maxPoint;
	pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);

	Eigen::Vector3f max_values;
	max_values.x() = std::abs(maxPoint.x - minPoint.x);
	max_values.y() = std::abs(maxPoint.y - minPoint.y);
	max_values.z() = std::abs(maxPoint.z - minPoint.z);

	if (max_values.maxCoeff() < this->magnitude_threshold)
		return true;
	else
		return false;
}

vector<int> GroundFilter::validate_clusters_by_ratio()
{
	vector<int> valid_clusters;
	valid_clusters.clear();
	pcl::IndicesPtr current_cluster(new pcl::Indices);

	int clust_indx = 0;
	for (auto cluster : this->regrow_clusters)
	{
		*current_cluster = cluster.indices;

		if (this->valid_ratio(current_cluster))
			valid_clusters.push_back(clust_indx);

		clust_indx++;
	}

	return valid_clusters;
}

vector<int> GroundFilter::validate_clusters_by_magnitude()
{
	vector<int> valid_clusters;
	valid_clusters.clear();

	PointCloud::Ptr tmp_cloud(new PointCloud);
	PointCloud::Ptr current_cluster_cloud(new PointCloud);
	pcl::IndicesPtr current_cluster(new pcl::Indices);

	int clust_indx = 0;
	for (auto cluster : this->regrow_clusters)
	{
		*current_cluster = cluster.indices;

		if (this->valid_magnitude(current_cluster))
			valid_clusters.push_back(clust_indx);

		clust_indx++;
	}

	return valid_clusters;
}

vector<int> GroundFilter::validate_clusters_hybrid()
{
	// cout << GREEN <<"Checking regrow clusters..." << RESET << endl;
	vector<int> valid_clusters;
	valid_clusters.clear();
	PointCloud::Ptr current_cluster_cloud(new PointCloud);
	PointCloud::Ptr remain_input_cloud(new PointCloud);
	pcl::IndicesPtr current_cluster(new pcl::Indices);

	int clust_indx = 0;
	for (auto cluster : this->regrow_clusters)
	{
		*current_cluster = cluster.indices;

		if (this->valid_magnitude(current_cluster) && this->valid_ratio(current_cluster))
			valid_clusters.push_back(clust_indx);

		clust_indx++;
	}
	return valid_clusters;
}

void GroundFilter::compute_metrics()
{

	this->cons.debug("Computing metrics...");
	auto start = std::chrono::high_resolution_clock::now();

	this->cm = utils::compute_conf_matrix(this->gt_truss_idx, this->gt_ground_idx, this->truss_idx, this->ground_idx);

	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	this->metrics_time = duration.count();
}

void GroundFilter::view_final_segmentation()
{
	this->getConfMatrixIndexes();
	PointCloud::Ptr error_cloud(new PointCloud);
	PointCloud::Ptr truss_cloud(new PointCloud);
	PointCloud::Ptr ground_cloud(new PointCloud);
	pcl::IndicesPtr error_idx(new pcl::Indices);

	error_idx->insert(error_idx->end(), this->fp_idx->begin(), this->fp_idx->end());
	error_idx->insert(error_idx->end(), this->fn_idx->begin(), this->fn_idx->end());

	truss_cloud = utils::extract_indices(cloud_in, this->tp_idx, false);
	ground_cloud = utils::extract_indices(cloud_in, this->tn_idx, false);
	error_cloud = utils::extract_indices(cloud_in, error_idx, false);

	pcl::visualization::PCLVisualizer my_vis;
	my_vis.setBackgroundColor(1, 1, 1);

	fs::path camera_params_path = "/home/fran/workSpaces/arvc_ws/src/" + this->cloud_id + "_camera_params.txt";

	try
	{
		my_vis.loadCameraParameters(camera_params_path.string());
		// my_vis.loadCameraParameters("/home/arvc/workSpaces/code_ws/build/" + this->cloud_id + "_camera_params.txt");
	}
	catch (const std::exception &e)
	{
	}

	pcl::visualization::PointCloudColorHandlerCustom<PointT> truss_color(truss_cloud, 50, 190, 50);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> ground_color(ground_cloud, 100, 100, 100);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> error_color(error_cloud, 200, 10, 10);

	my_vis.addPointCloud(truss_cloud, truss_color, "truss_cloud");
	my_vis.addPointCloud(ground_cloud, ground_color, "ground_cloud");
	my_vis.addPointCloud(error_cloud, error_color, "error_cloud");
	my_vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "truss_cloud");
	my_vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "ground_cloud");
	my_vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "error_cloud");

	my_vis.addCoordinateSystem(0.8, "sensor_origin");
	auto pos = cloud_in->sensor_origin_;
	auto ori = cloud_in->sensor_orientation_;

	while (!my_vis.wasStopped())
	{
		my_vis.saveCameraParameters(camera_params_path.string());
		my_vis.spinOnce(100);
	}
}

void GroundFilter::getConfMatrixIndexes()
{
	// For the index points classified as truss, check if this point is in the ground truth truss. If it is, it is a true positive, otherwise it is a false positive.
	for (size_t i = 0; i < this->truss_idx->size(); i++)
	{
		if (std::find(this->gt_truss_idx->begin(), this->gt_truss_idx->end(), this->truss_idx->at(i)) != this->gt_truss_idx->end())
			this->tp_idx->push_back(this->truss_idx->at(i));
		else
			this->fp_idx->push_back(this->truss_idx->at(i));
	}

	// For the index points classified as ground, check if this point is in the ground truth ground. If it is, it is a true negative, otherwise it is a false negative.
	for (size_t i = 0; i < this->ground_idx->size(); i++)
	{
		if (std::find(this->gt_ground_idx->begin(), this->gt_ground_idx->end(), this->ground_idx->at(i)) != this->gt_ground_idx->end())
			this->tn_idx->push_back(this->ground_idx->at(i));
		else
			this->fn_idx->push_back(this->ground_idx->at(i));
	}
}

ConfusionMatrixIndexes GroundFilter::getConfMatrixIndexes(pcl::IndicesPtr &_pred_truss_idx, pcl::IndicesPtr &_pred_ground_idx) {
	ConfusionMatrixIndexes cm_indexes;

	// For the index points classified as truss, check if this point is in the ground truth truss. If it is, it is a true positive, otherwise it is a false positive.
	for (size_t i = 0; i < _pred_truss_idx->size(); i++)
	{
		if (std::find(this->gt_truss_idx->begin(), this->gt_truss_idx->end(), _pred_truss_idx->at(i)) != this->gt_truss_idx->end())
			cm_indexes.tp->push_back(_pred_truss_idx->at(i));
		else
			cm_indexes.fp->push_back(_pred_truss_idx->at(i));
	}

	// For the index points classified as ground, check if this point is in the ground truth ground. If it is, it is a true negative, otherwise it is a false negative.
	for (size_t i = 0; i < _pred_ground_idx->size(); i++)
	{
		if (std::find(this->gt_ground_idx->begin(), this->gt_ground_idx->end(), _pred_ground_idx->at(i)) != this->gt_ground_idx->end())
			cm_indexes.tn->push_back(_pred_ground_idx->at(i));
		else
			cm_indexes.fn->push_back(_pred_ground_idx->at(i));
	}

	return cm_indexes;
}


// Public methods
GroundFilter::GroundFilter()
{

	this->cloud_in = PointCloud::Ptr(new PointCloud);
	this->cloud_out = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
	this->cloud_in_labeled = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);

	this->coarse_ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->coarse_truss_idx = pcl::IndicesPtr(new pcl::Indices);

	this->truss_idx = pcl::IndicesPtr(new pcl::Indices);
	this->ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->gt_truss_idx = pcl::IndicesPtr(new pcl::Indices);
	this->gt_ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->low_density_idx = pcl::IndicesPtr(new pcl::Indices);
	this->wrong_idx = pcl::IndicesPtr(new pcl::Indices);
	this->tp_idx = pcl::IndicesPtr(new pcl::Indices);
	this->fp_idx = pcl::IndicesPtr(new pcl::Indices);
	this->fn_idx = pcl::IndicesPtr(new pcl::Indices);
	this->tn_idx = pcl::IndicesPtr(new pcl::Indices);

	this->cons.enable_vis = false;
	this->enable_metrics = false;
	this->normals_time = 0;
	this->metrics_time = 0;
	this->ratio_threshold = 0.3f;
	this->magnitude_threshold = 1000.0f;
	this->sac_threshold = 0.5f;
	this->mode = MODE::HYBRID;
	this->cloud_id = "NO_ID";
	this->save_cloud = false;
	this->save_cloud_path = "./inferences/";
}

GroundFilter::GroundFilter(YAML::Node _cfg)
{
	this->cfg = _cfg;
	this->cloud_in = PointCloud::Ptr(new PointCloud);
	this->cloud_out = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
	this->cloud_in_labeled = pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);

	this->coarse_ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->coarse_truss_idx = pcl::IndicesPtr(new pcl::Indices);

	this->truss_idx = pcl::IndicesPtr(new pcl::Indices);
	this->ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->gt_truss_idx = pcl::IndicesPtr(new pcl::Indices);
	this->gt_ground_idx = pcl::IndicesPtr(new pcl::Indices);
	this->low_density_idx = pcl::IndicesPtr(new pcl::Indices);
	this->wrong_idx = pcl::IndicesPtr(new pcl::Indices);
	this->tp_idx = pcl::IndicesPtr(new pcl::Indices);
	this->fp_idx = pcl::IndicesPtr(new pcl::Indices);
	this->fn_idx = pcl::IndicesPtr(new pcl::Indices);
	this->tn_idx = pcl::IndicesPtr(new pcl::Indices);

	this->normals_time = 0;
	this->metrics_time = 0;

	this->cons.enable = _cfg["EN_DEBUG"].as<bool>();
	this->cons.enable_vis = _cfg["EN_VISUAL"].as<bool>();
	this->enable_metrics = _cfg["EN_METRIC"].as<bool>();

	this->node_length = _cfg["NODE_LENGTH"].as<float>();
	this->node_width = _cfg["NODE_WIDTH"].as<float>();
	this->sac_threshold = _cfg["SAC_THRESHOLD"].as<float>();
	this->voxel_size = _cfg["VOXEL_SIZE"].as<float>();

	this->density_first = _cfg["DENSITY"]["first"].as<bool>();
	this->enable_density_filter = _cfg["DENSITY"]["enable"].as<bool>();
	this->density_radius = _cfg["DENSITY"]["radius"].as<float>();
	this->density_threshold = _cfg["DENSITY"]["threshold"].as<int>();

	this->enable_euclidean_clustering = false; //_cfg["EUCLID"]["enable"].as<bool>();
	// this->cluster_radius = _cfg["EUCLID"]["radius"].as<float>();
	// this->cluster_min_size = _cfg["EUCLID"]["min_size"].as<int>();

	this->mode = parse_MODE(_cfg["MODE"].as<std::string>());
	this->save_cloud = _cfg["SAVE_CLOUD"].as<bool>();
	this->save_cloud_path = _cfg["SAVE_CLOUD_PATH"].as<std::string>();

	this->cons.debug("GroundFilter configured in mode: " + parse_MODE(this->mode), "GREEN");

	this->cloud_id = "NO_ID";
}

GroundFilter::~GroundFilter()
{
}

void GroundFilter::set_input_cloud(pcl::PointCloud<pcl::PointXYZL>::Ptr &_cloud)
{

	utils::GtIndices tmp_gt = utils::get_ground_truth_indices(_cloud);
	this->gt_truss_idx = tmp_gt.truss;
	this->gt_ground_idx = tmp_gt.ground;

	pcl::copyPointCloud(*_cloud, *this->cloud_in);
}

void GroundFilter::set_mode(MODE _mode)
{
	this->mode = _mode;
}

void GroundFilter::set_node_length(float _length)
{
	this->node_length = _length;
}

void GroundFilter::set_node_width(float _width)
{
	this->node_width = _width;
}

void GroundFilter::set_sac_threshold(float _threshold)
{
	this->sac_threshold = _threshold;
}

void GroundFilter::set_voxel_size(float _voxel_size)
{
	this->voxel_size = _voxel_size;
}

void GroundFilter::save_cloud_result()
{

	if (!fs::exists(this->save_cloud_path))
		fs::create_directories(this->save_cloud_path);

	// SAVE THE CLOUD RESULT
	pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZL>);
	pcl::copyPointCloud(*this->cloud_in, *cloud_out);

	for (int i = 0; i < cloud_out->points.size(); i++)
	{

		auto it = std::find(this->truss_idx->begin(), this->truss_idx->end(), i);

		if (it != this->truss_idx->end())
			cloud_out->points[i].label = 1;
		else
			cloud_out->points[i].label = 0;
	}

	pcl::PLYWriter writer;
	std::stringstream ss;
	ss.str("");
	std::string mode = parse_MODE(this->mode);
	ss << this->save_cloud_path.string() << mode << "_" << this->cloud_id << "_inf.ply";
	writer.write(ss.str(), *cloud_out);
	this->cons.debug("Cloud saved in: " + ss.str(), "GREEN");
}

float GroundFilter::get_ground_data_ratio()
{
	return (float)this->gt_ground_idx->size() / ((float)this->gt_ground_idx->size() + (float)this->gt_truss_idx->size());
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr GroundFilter::get_regrow_output(){

	pcl::PointCloud<pcl::PointXYZRGB>::Ptr regrow_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

	for (size_t i = 0; i < this->regrow_clusters.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
		
		uint8_t r = rand() % 256;
		uint8_t g = rand() % 256;
		uint8_t b = rand() % 256;

		for (size_t j = 0; j < this->regrow_clusters[i].indices.size(); j++)
		{
			pcl::PointXYZRGB point;
			point.x = this->cloud_in->points[this->regrow_clusters[i].indices[j]].x;
			point.y = this->cloud_in->points[this->regrow_clusters[i].indices[j]].y;
			point.z = this->cloud_in->points[this->regrow_clusters[i].indices[j]].z;
			point.r = r;
			point.g = g;	
			point.b = b;
			cluster_cloud->points.push_back(point);
		}

		*regrow_cloud += *cluster_cloud;
	}

	return regrow_cloud;
}


pcl::PointCloud<pcl::PointXYZRGB>::Ptr GroundFilter::get_coarse_segmentation()
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr coarse_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*this->cloud_in, *coarse_cloud);

	for (size_t i = 0; i < coarse_cloud->points.size(); i++)
	{
		auto it_truss = std::find(this->coarse_truss_idx->begin(), this->coarse_truss_idx->end(), i);
		auto it_ground = std::find(this->coarse_ground_idx->begin(), this->coarse_ground_idx->end(), i);

		if (it_truss != this->coarse_truss_idx->end())
		{
			coarse_cloud->points[i].r = 100;
			coarse_cloud->points[i].g = 100;
			coarse_cloud->points[i].b = 100;
		}
		else if (it_ground != this->coarse_ground_idx->end())
		{
			coarse_cloud->points[i].r = 200;
			coarse_cloud->points[i].g = 200;
			coarse_cloud->points[i].b = 50;
		}
		else
		{
			coarse_cloud->points[i].r = 200;
			coarse_cloud->points[i].g = 10;
			coarse_cloud->points[i].b = 10;
		}

	}

	return coarse_cloud;
}
int GroundFilter::compute()
{
	switch (this->mode)
	{
	case MODE::RATIO:
		this->ratio_threshold = this->node_width / this->sac_threshold;
		this->coarse_segmentation();
		this->fine_segmentation();
		this->update_segmentation();

		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::MAGNITUDE:
		this->magnitude_threshold = this->sac_threshold;
		this->coarse_segmentation();
		this->fine_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::HYBRID:
		this->ratio_threshold = this->node_width / this->sac_threshold;
		this->magnitude_threshold = this->sac_threshold;
		this->coarse_segmentation();
		this->fine_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::WOCOARSE_RATIO:
		this->ratio_threshold = this->node_width / this->node_length;
		this->fine_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::WOCOARSE_MAGNITUDE:
		this->magnitude_threshold = this->node_length;
		this->fine_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::WOCOARSE_HYBRID:
		this->ratio_threshold = this->node_width / this->node_length;
		this->magnitude_threshold = this->node_length;
		this->fine_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	case MODE::WOFINE:
		this->coarse_segmentation();
		this->update_segmentation();
		if (this->enable_density_filter and this->enable_euclidean_clustering)
			std::cout << "ERROR: Density filter and Euclidean clustering cannot be enabled at the same time" << std::endl;
		else if (this->enable_density_filter)
			this->density_filter();
		else if (this->enable_euclidean_clustering)
			this->euclidean_clustering();
		break;

	default:
		this->cons.debug("ERROR: Invalid mode selected, options are: RATIO, MAGNITUDE, HYBRID, WOFINE, WOCOARSE_RATIO, WOCOARSE_MAGNITUDE, WOCOARSE_HYBRID", "RED");
		break;
	}

	if (this->save_cloud) {
		this->cons.debug("Trying to save cloud result", "GREEN");
		this->save_cloud_result();
	}

	if (this->cfg["VIEW_FINAL_SEGMENTATION"].as<bool>()){
		this->cons.debug("Trying to visualizate results", "GREEN");
		this->view_final_segmentation();
	}

	if (this->enable_metrics)
	{
		this->cons.debug("Trying to compute metrics", "GREEN");
		this->compute_metrics();
	}

	return 0;
}
