/* USAGE:
    * 0. Compile the package with <catkin_make> or <cmake .. && make>
    * 1. Set config.yaml to desired parameters: Mode, Node Length, Node Width, Sac Threshold, Voxel Size
    * 2. Go to the root dir where the clouds are located
    * 3. Run the following command:
    * if compiled with catkin_make (ROS)  
    *   rosrun analytical_rs_seg ground_filter_node <path_to_cloud> <mode>{ratio, magnitude, hybrid, wofine, wocoarse_ratio, wocoarse_magnitude, wocoarse_hybrid}
    * else if compiled with cmake
    *   add build folder to the env variable PATH 
    *   run: ground_filter_node <path_to_cloud> <mode>{ratio, magnitude, hybrid, wofine, wocoarse_ratio, wocoarse_magnitude, wocoarse_hybrid}
    *   !! If <path_to_cloud> is not set, will apply the algorithm to every cloud found in the current directory !!
    * 
    * Name: Fran Soler Mora
    * email: f.soler@umh.es
 */


#include "analytical_rs_seg/ground_filter.hpp"


std::vector<fs::path> get_data_paths(int argc, char **argv)
{
    std::vector<fs::path> path_vector;
    
    // COMPUTE THE ALGORITHM FOR EVERY CLOUD IN THE CURRENT FOLDER
    if (argc == 1)
    {
        fs::path current_dir = fs::current_path();

        // Save all the paths of the clouds in the current directory for the tqdm loop
        for (const auto &entry : fs::directory_iterator(current_dir))
        {
            if (entry.path().extension() == ".pcd" || entry.path().extension() == ".ply")
                path_vector.push_back(entry.path());
        }
    }
    // COMPUTE THE ALGORITHM ONLY ONE CLOUD PASSED AS ARGUMENT IN CURRENT FOLDER
    else if (argc > 1)
    {
        fs::path entry = argv[1];
        std::cout << "Processing cloud: " << entry << std::endl;

        if (entry.extension() == ".pcd" || entry.extension() == ".ply")
        {
            path_vector.push_back(entry);
        }
    }
    else
    {
        std::cout << "\tNo mode selected." << std::endl;
        std::cout << "\tUsage:" << std::endl;
        std::cout << "\tAdd build folder to the env variable PATH" << std::endl;
        std::cout << "\tRun ground_filter_node <path_to_cloud> <mode>{ratio, magnitude, hybrid, wofine, wocoarse}" << std::endl;
    }

    return path_vector;
}

// Function to get memory usage in bytes
long getMemoryUsageInBytes() {
    std::ifstream statm_file("/proc/self/status");
    std::string line;
    long memoryUsage = 0;

    while (std::getline(statm_file, line)) {
        if (line.find("VmRSS:") != std::string::npos) {  // VmRSS gives resident memory in KB
            std::istringstream iss(line);
            std::string key;
            long value;
            std::string unit;
            iss >> key >> value >> unit;  // Extract key, value, and unit (should be "kB")
            memoryUsage = value * 1024;   // Convert from kilobytes to bytes
            break;
        }
    }
    return memoryUsage;  // Return memory in bytes
}


void best_density_estimation(std::vector<fs::path> path_vector, YAML::Node config)
{

    #include <map>
    typedef pcl::PointXYZLNormal PointIN;

    // VARIABLES UTILITIES
    utils::Metrics global_metrics;

    pcl::PointCloud<PointIN>::Ptr input_cloud_xyzln (new pcl::PointCloud<PointIN>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    const int dataset_size = path_vector.size();

    std::map<int, float> threshold_f1_map;
    
    const int START_DENSITY_THRESHOLD = config["START_DENSITY_THRESHOLD"].as<int>();
    const int STEP_DENSITY_THRESHOLD = config["STEP_DENSITY_THRESHOLD"].as<int>();

    int current_threshold = START_DENSITY_THRESHOLD;

    // BUSQUEDA DEL MEJOR THRESHOLD
    for (int i=0 ;  i < config["ITERATIONS"].as<int>(); i++) {

        if (i > 0)
            current_threshold = current_threshold + STEP_DENSITY_THRESHOLD;


        std::cout << std::endl <<"CURRENT THRESHOLD: " << current_threshold << std::endl;
        for (const fs::path &entry : tq::tqdm(path_vector))
        {
            input_cloud_xyzln = utils::readPointCloud<PointIN>(entry);
            pcl::copyPointCloud(*input_cloud_xyzln, *cloud);

            config["DENSITY"]["threshold"] = current_threshold;
            // config["CROP_SET"] = 100;

            GroundFilter gf(config);

            gf.cloud_id = entry.stem();
            gf.set_input_cloud(cloud);
            gf.compute();
            
            if (config["EN_METRIC"].as<bool>())
            {
                global_metrics.tp += gf.cm.tp;
                global_metrics.tn += gf.cm.tn;
                global_metrics.fp += gf.cm.fp;
                global_metrics.fn += gf.cm.fn;
            }
        }

        // global_metrics.plotMetrics();
        threshold_f1_map[current_threshold] = global_metrics.f1_score();

    }

    std::cout << "Threshold F1 Map" << config["DENSITY"]["radius"] << ": " << std::endl;
    for (auto const& [key, val] : threshold_f1_map)
    {
        std::cout << key << " : " << val << std::endl;
    }


}


void best_voxel_estimation(std::vector<fs::path> path_vector, YAML::Node config)
{

    typedef pcl::PointXYZLNormal PointIN;
    #include <map>

    // VARIABLES UTILITIES
    utils::Metrics global_metrics;

    pcl::PointCloud<PointIN>::Ptr input_cloud_xyzln (new pcl::PointCloud<PointIN>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    const int dataset_size = path_vector.size();

    std::map<float, float> voxel_miou_map;
    std::map<float, float> voxel_f1_map;
    
    const float START_VOXEL = config["START_VOXEL"].as<float>();
    const float STEP_VOXEL = config["STEP_VOXEL"].as<float>();
    float current_voxel = START_VOXEL;
    
    // BUSQUEDA DEL MEJOR THRESHOLD
    for (int i=0 ;  i < 10; i++) {

        if (i > 0)
            current_voxel = current_voxel + STEP_VOXEL;


        std::cout << "CURRENT VOXEL: " << current_voxel << std::endl;
        for (const fs::path &entry : tq::tqdm(path_vector))
        {
            input_cloud_xyzln = utils::readPointCloud<PointIN>(entry);
            pcl::copyPointCloud(*input_cloud_xyzln, *cloud);

            config["VOXEL_SIZE"] = current_voxel;
            // config["CROP_SET"] = 100;

            GroundFilter gf(config);

            gf.cloud_id = entry.stem();
            gf.set_input_cloud(cloud);
            gf.compute();
            
            if (config["EN_METRIC"].as<bool>())
            {
                global_metrics.tp += gf.cm.tp;
                global_metrics.tn += gf.cm.tn;
                global_metrics.fp += gf.cm.fp;
                global_metrics.fn += gf.cm.fn;
            }
        }

        // global_metrics.plotMetrics();
        voxel_miou_map[current_voxel] = global_metrics.miou();
        voxel_f1_map[current_voxel] = global_metrics.f1_score();

    }

    std::cout << "Voxel MIoU Map: " << std::endl;
    for (auto const& [key, val] : voxel_miou_map)
    {
        std::cout << key << " : " << val << std::endl;
    }

    std::cout << "Voxel F1 Map: " << std::endl;
    for (auto const& [key, val] : voxel_f1_map)
    {
        std::cout << key << " : " << val << std::endl;
    }

}


int main(int argc, char **argv)
{
    typedef pcl::PointXYZLNormal PointIN;
    

    pcl::console::setVerbosityLevel(pcl::console::L_ALWAYS);
    std::cout << YELLOW << "Running Ground Filter Node:" << RESET << std::endl;


    fs::path CONFIG = fs::path(PROJECT_PATH) / "config/config.yaml";

    YAML::Node config = YAML::LoadFile(CONFIG.string());
    
    std::vector<fs::path> path_vector;
    path_vector = get_data_paths(argc, argv);
    

    if (config["CROP_SET"].as<int>() != 0)
    {
        path_vector.resize(config["CROP_SET"].as<int>());
    }



    // VARIABLES UTILITIES
    utils::Metrics global_metrics;

    int normals_time = 0;
    int metrics_time = 0;

    float coarse_ground_size_ratio = 0.0;

    string best_recall_cloud;
    float best_recall = 0;

    pcl::PointCloud<PointIN>::Ptr input_cloud_xyzln (new pcl::PointCloud<PointIN>);
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    const int dataset_size = path_vector.size();


    if (argc > 1)
    {
        auto initial_memory = getMemoryUsageInBytes();
        auto start = std::chrono::high_resolution_clock::now();
        fs::path entry = argv[1];
        input_cloud_xyzln = utils::readPointCloud<PointIN>(entry);
        pcl::copyPointCloud(*input_cloud_xyzln, *cloud);

        GroundFilter gf(config);

        gf.cloud_id = entry.stem();
        gf.set_input_cloud(cloud);
        gf.compute();

        normals_time += gf.normals_time;
        metrics_time += gf.metrics_time;

        if (config["EN_METRIC"].as<bool>())
        {
            global_metrics.tp += gf.cm.tp;
            global_metrics.tn += gf.cm.tn;
            global_metrics.fp += gf.cm.fp;
            global_metrics.fn += gf.cm.fn;
        }

        auto final_memory = getMemoryUsageInBytes();

        std::cout << "\n\nMETRICS: " << std::endl;

        global_metrics.plotMetrics();

        // PRINT COMPUTATION TIME
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        int avg_time = (int)floor((duration.count() - metrics_time) / dataset_size);
        std::cout << "\tAvg. Computation Time: " << avg_time << " ms" << std::endl;

        long avg_memory = final_memory - initial_memory;
        std::cout << "\tAvg. Memory Usage: " << avg_memory << " bytes" << std::endl;

        return 0;
    }

    // best_voxel_estimation(path_vector, config);
    // best_density_estimation(path_vector,config);

    std::cout << "\t Evaluating clouds: " << path_vector.size() << std::endl;
    std::vector<long> memory_vector;
    auto start = std::chrono::high_resolution_clock::now();
    for (const fs::path &entry : tq::tqdm(path_vector))
    {

        auto initial_memory = getMemoryUsageInBytes();
        input_cloud_xyzln = utils::readPointCloud<PointIN>(entry);
        pcl::copyPointCloud(*input_cloud_xyzln, *cloud);

        GroundFilter gf(config);

        gf.cloud_id = entry.stem();
        gf.set_input_cloud(cloud);
        gf.compute();
        
        normals_time += gf.normals_time;
        metrics_time += gf.metrics_time;

        if (config["EN_METRIC"].as<bool>())
        {
            global_metrics.tp += gf.cm.tp;
            global_metrics.tn += gf.cm.tn;
            global_metrics.fp += gf.cm.fp;
            global_metrics.fn += gf.cm.fn;
        }
        auto final_memory = getMemoryUsageInBytes();
        memory_vector.push_back(final_memory - initial_memory);
    }

    std::cout << "\n\nMETRICS: " << std::endl;

    global_metrics.plotMetrics();

    // PRINT COMPUTATION TIME
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    int avg_time = (int)floor((duration.count() - metrics_time) / dataset_size);
    std::cout << "\tAvg. Computation Time: " << avg_time << " ms" << std::endl;

    long avg_memory = std::accumulate(memory_vector.begin(), memory_vector.end(), 0) / memory_vector.size();
    std::cout << "\tAvg. Memory Usage: " << avg_memory << " bytes" << std::endl;

    return 0;
}

