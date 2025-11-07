/* USAGE:
    * 1. Set config.yaml to desired parameters: Mode, Node Length, Node Width, Sac Threshold, Voxel Size
    * 2. Go to the root dir where the clouds are located
    * 3. Run the following command:  
    * rosrun arvc_ground_filter ground_filter_node <path_to_cloud> <mode>{ratio, module, hybrid, wofine, wocoarse_ratio, wocoarse_module, wocoarse_hybrid}
    *   !! If <path_to_cloud> is not set, will apply the algorithm to every cloud found in the current directory !!
    * 
    * Name: Fran Soler Mora
    * email: f.soler@umh.es
 */

#include "analytical_rs_seg/ground_filter.hpp"
#include <yaml-cpp/yaml.h>
#include <cstdlib> // For std::getenv
#include <pcl/common/common.h>
#include "utils.hpp"
#include "csv_utils.hpp"
// #include <ros/package.h>


typedef pcl::PointXYZLNormal PointIN;

std::vector<fs::path> get_data_paths(fs::path root_dir)
{
    std::vector<fs::path> path_vector;

    root_dir = root_dir / "ply_xyzln";

    for (const auto &entry : fs::directory_iterator(root_dir))
    {
        if (entry.path().extension() == ".ply")
            path_vector.push_back(entry.path());
    }
    return path_vector;
}


void find_differences(YAML::Node config, std::vector<fs::path> path_vector){

    std::string set = path_vector[0].parent_path().parent_path().filename().string();
    std::cout << "EXP MODE:  " << config["MODE"].as<string>() << " -- DATASET: " << set << std::endl;
    // VARIABLES UTILITIES
    utils::Metrics global_metrics;
    csv_data data;  

    int normals_time = 0;
    int metrics_time = 0;
    int dataset_size = path_vector.size();
    int ground_size = 0;
    int truss_size = 0;

    float coarse_ground_size_ratio = 0.0;

    string set_id = path_vector[0].parent_path().parent_path().filename().string();
    string best_recall_cloud;
    float best_recall = 0;

    pcl::PointCloud<PointIN>::Ptr input_cloud_xyzln (new pcl::PointCloud<PointIN>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_hyb (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_mag (new pcl::PointCloud<pcl::PointXYZL>);


    std::ofstream file;
    fs::path current_dir = fs::current_path();
    file.open(current_dir.string() + "/mag_vs_hybrid_different_clouds.txt");
    // tqdm loop
    auto start = std::chrono::high_resolution_clock::now();
    for (const fs::path &entry : path_vector)
    {
        input_cloud_xyzln = utils::readPointCloud<PointIN>(entry);

        // GROUND_FILTER_HYBRID
        pcl::copyPointCloud(*input_cloud_xyzln, *cloud_hyb);
        config["MODE"] = "hybrid";
        GroundFilter gf_hyb(config);

        gf_hyb.set_input_cloud(cloud_hyb);
        gf_hyb.cloud_id = entry.stem().string();
        gf_hyb.dataset_id = set_id;
        gf_hyb.compute();
        
        // normals_time += gf_hyb.normals_time;
        // metrics_time += gf_hyb.metrics_time;
        // ground_size += gf_hyb.gt_ground_idx->size();
        // truss_size += gf_hyb.gt_truss_idx->size();

        // GROUND_FILTER_MAGNITUDE
        pcl::copyPointCloud(*input_cloud_xyzln, *cloud_mag);
        config["MODE"] = "magnitude";
        GroundFilter gf_mag(config);

        gf_mag.set_input_cloud(cloud_mag);
        gf_mag.cloud_id = entry.stem().string();
        gf_mag.dataset_id = set_id;
        gf_mag.compute();
        
        // normals_time += gf_mag.normals_time;
        // metrics_time += gf_mag.metrics_time;
        // ground_size += gf_mag.gt_ground_idx->size();
        // truss_size += gf_mag.gt_truss_idx->size();


        // COMPARE CONFUSION MATRICES
        if (gf_hyb.cm == gf_mag.cm) {
            // std::cout << "CLOUD " << entry.stem().string() << ": EQUAL RESULTS" << std::endl;
            continue;
        } else {
            std::cout << "########### -> CLOUD " << entry.stem().string() << ": DIFFERENT RESULTS" << std::endl;

            file << entry.string() << std::endl;
        }
    }
    file.close();

}


void view_differences(YAML::Node config, const fs::path& cloud_path){

    pcl::PointCloud<PointIN>::Ptr input_cloud_xyzln (new pcl::PointCloud<PointIN>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_hyb (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_mag (new pcl::PointCloud<pcl::PointXYZL>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr mag_seg (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr hyb_seg (new pcl::PointCloud<pcl::PointXYZRGB>);


    input_cloud_xyzln = utils::readPointCloud<PointIN>(cloud_path);

    // GROUND_FILTER_HYBRID
    pcl::copyPointCloud(*input_cloud_xyzln, *cloud_hyb);
    config["MODE"] = "hybrid";
    GroundFilter gf_hyb(config);

    gf_hyb.set_input_cloud(cloud_hyb);
    gf_hyb.cloud_id = cloud_path.stem().string();
    // gf_hyb.dataset_id = "";
    gf_hyb.compute();

    ConfusionMatrixIndexes cm_hyb = gf_hyb.getConfMatrixIndexes(gf_hyb.truss_idx, gf_hyb.ground_idx);

    // Build HYB SEGMENTATION CLOUD rgb
    for (size_t idx =0; idx < cloud_hyb->points.size(); idx++) {
        pcl::PointXYZRGB point;
        point.x = cloud_hyb->points[idx].x;
        point.y = cloud_hyb->points[idx].y;
        point.z = cloud_hyb->points[idx].z;


        if (std::find(cm_hyb.tp->begin(), cm_hyb.tp->end(), idx) != cm_hyb.tp->end())
        {
            point.r = 0;
            point.g = 255;
            point.b = 0;
            hyb_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_hyb.tn->begin(), cm_hyb.tn->end(), idx) != cm_hyb.tn->end())
        {
            point.r = 100;
            point.g = 100;
            point.b = 100;
            hyb_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_hyb.fp->begin(), cm_hyb.fp->end(), idx) != cm_hyb.fp->end())
        {
            point.r = 0;
            point.g = 0;
            point.b = 255;
            hyb_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_hyb.fn->begin(), cm_hyb.fn->end(), idx) != cm_hyb.fn->end())
        {
            point.r = 255;
            point.g = 0;
            point.b = 0;
            hyb_seg->points.push_back(point);
            continue;
        }
        hyb_seg->points.push_back(point);
    }

    // GROUND_FILTER_MAGNITUDE
    pcl::copyPointCloud(*input_cloud_xyzln, *cloud_mag);
    config["MODE"] = "magnitude";
    GroundFilter gf_mag(config);

    gf_mag.set_input_cloud(cloud_mag);
    gf_mag.cloud_id = cloud_path.stem().string();
    // gf_mag.dataset_id = set_id;
    gf_mag.compute();
    ConfusionMatrixIndexes cm_mag = gf_mag.getConfMatrixIndexes(gf_mag.truss_idx, gf_mag.ground_idx);

    // Build MAG SEGMENTATION CLOUD rgb
    for (size_t idx =0; idx < cloud_mag->points.size(); idx++) {
        pcl::PointXYZRGB point;
        point.x = cloud_mag->points[idx].x;
        point.y = cloud_mag->points[idx].y;
        point.z = cloud_mag->points[idx].z;

        if (std::find(cm_mag.tp->begin(), cm_mag.tp->end(), idx) != cm_mag.tp->end())
        {
            point.r = 0;
            point.g = 255;
            point.b = 0;
            mag_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_mag.tn->begin(), cm_mag.tn->end(), idx) != cm_mag.tn->end())
        {
            point.r = 100;
            point.g = 100;
            point.b = 100;
            mag_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_mag.fp->begin(), cm_mag.fp->end(), idx) != cm_mag.fp->end())
        {
            point.r = 0;
            point.g = 0;
            point.b = 255;
            mag_seg->points.push_back(point);
            continue;
        }
        if (std::find(cm_mag.fn->begin(), cm_mag.fn->end(), idx) != cm_mag.fn->end())
        {
            point.r = 255;
            point.g = 0;
            point.b = 0;
            mag_seg->points.push_back(point);
            continue;
        }
        mag_seg->points.push_back(point);
    }

    // IF DIFFERENT, VIEW
    if (!(gf_hyb.cm == gf_mag.cm)) {

        std::cout << "Report: " << std::endl;
        std::cout << "\t         HYBRID     ---     MAGNITUDE" << std::endl;
        std::cout << "\tTP: " << gf_hyb.cm.tp << "     ---     " << gf_mag.cm.tp << std::endl;
        std::cout << "\tTN: " << gf_hyb.cm.tn << "     ---     " << gf_mag.cm.tn << std::endl;
        std::cout << "\tFP: " << gf_hyb.cm.fp << "     ---     " << gf_mag.cm.fp << std::endl;
        std::cout << "\tFN: " << gf_hyb.cm.fn << "     ---     " << gf_mag.cm.fn << std::endl;


        pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("Difference Viewer"));
        int v1(0);
        int v2(1);
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);
        viewer->setBackgroundColor(1,1,1);
        viewer->addText("HYBRID", 10, 10, "v1 text", v1);
        viewer->addText("MAGNITUDE", 10, 10, "v2 text", v2);

        viewer->addPointCloud<pcl::PointXYZRGB>(hyb_seg, "hyb_seg", v1);
        viewer->addPointCloud<pcl::PointXYZRGB>(mag_seg, "mag_seg", v2);

        while (!viewer->wasStopped ())
        {
            viewer->spinOnce (100);
        }


    }
}



int main(int argc, char **argv)
{
    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    std::cout << YELLOW << "Running Ground Filter Node:" << RESET << std::endl;


    const char *homeDir = std::getenv("HOME");
    const string HOME = homeDir;
    const fs::path CONFIG = HOME + "/workspaces/arvc_ws/src/rs_seg_methods/analytical_rs_seg/config/config.yaml";
    const fs::path ROOT_DIR = HOME + "/datasets/arvc_truss/test";

    // const std::vector<std::string> DATASETS = {"orto", "crossed", "00", "01", "02", "03"};
    const std::vector<std::string> DATASETS = {"orthogonal", "crossed"};
    const std::vector<std::string> MODES = {"ratio", "module", "hybrid", "wofine", "wocoarse_ratio", "wocoarse_module", "wocoarse_hybrid"};

    
    YAML::Node config = YAML::LoadFile(CONFIG.string());

    std::vector<fs::path> path_vector;

    path_vector = get_data_paths(ROOT_DIR / DATASETS[0]);

    if (config["CROP_SET"].as<int>() != 0)
    {
        path_vector.resize(config["CROP_SET"].as<int>());
    }


    std::ifstream diff_file;
    fs::path current_dir = fs::current_path();
    diff_file.open(current_dir.string() + "/mag_vs_hybrid_different_clouds.txt");
    std::string cloud_line;

    std::vector<fs::path> diff_clouds;

    while (std::getline(diff_file, cloud_line))
    {
        fs::path cloud_path(cloud_line);
        diff_clouds.push_back(cloud_path);
    }
    diff_file.close();


    // find_differences(config, path_vector);

    for (const fs::path &cloud_path : diff_clouds)
        view_differences(config, cloud_path);

    return 0;
}


