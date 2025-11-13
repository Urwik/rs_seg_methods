#ifndef CSV_UTILS_HPP
#define CSV_UTILS_HPP

#include <iostream>
#include <fstream>
#include <istream>
#include <vector>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;


    std::string csv_header = "EXPERIMENT_ID,MODE,SET,NODE_LENGTH,NODE_WIDTH,SAC_THRESHOLD,VOXEL_SIZE,RAT_TH,MAG_TH,CROP_SET,SET_SIZE,PRECISION,RECALL,F1_SCORE,ACCURACY,MIOU,TP,TN,FP,FN,EXEC_TIME,GROUND_SIZE,TRUSS_SIZE,ENABLE_DENSITY,DENSITY_FIRST,DENSITY_RADIUS,DENSITY_THRESHOLD,ENABLE_EUCLID,EUCLID_RADIUS,EUCLID_MIN_SIZE";

struct csv_data
{
    string experiment_id = "none";
    string set_id = "none";
    int set_size = 0;
    float precision = 0, recall = 0, f1_score = 0, accuracy = 0, miou = 0;
    int tp = 0, tn = 0, fp = 0, fn = 0;
    int exec_time = 0;
    int ground_size = 0;
    int truss_size = 0;
    
    std::string mode;
    float node_length = 0;
    float node_width = 0;
    float sac_threshold = 0;
    float voxel_size = 0;
    int crop_set = 0;
    
    bool density_first = false;
    bool density_enable = false;
    float density_radius = 0;
    int density_threshold = 0;
    
    bool euclid_enable = false;
    float euclid_radius = 0;
    int euclid_min_size = 0;

    float magnitude_threshold = 0;
    float ratio_threshold = 0;
};



void writeToCSV(const fs::path& dir_path, const csv_data& data) {
    
    if (!fs::exists(dir_path)) {
        fs::create_directories(dir_path);
    }

    fs::path csv_path = dir_path / "test.csv";
    
    
    std::ifstream inFile(csv_path);
    bool isEmpty = inFile.peek() == std::ifstream::traits_type::eof();
    inFile.close();


    std::ofstream csv_file; 
    csv_file.open(csv_path, std::ios::app | std::ios::out);

    // Check if the file is open
    if (!csv_file.is_open()) {
        std::cerr << "Error opening file " << csv_path << std::endl;
        return;
    }

    else if (isEmpty) {
        csv_file << csv_header << "\n";
    }

    csv_file << data.experiment_id << ",";
    csv_file << data.mode << ",";
    csv_file << data.set_id << ",";

    csv_file << data.node_length << ",";
    csv_file << data.node_width << ",";
    csv_file << data.sac_threshold << ",";
    csv_file << data.voxel_size << ",";
    csv_file << data.ratio_threshold << ",";
    csv_file << data.magnitude_threshold << ",";
    csv_file << data.crop_set << ",";

    csv_file << data.set_size << ",";
    csv_file << data.precision << ",";
    csv_file << data.recall << ",";
    csv_file << data.f1_score << ",";
    csv_file << data.accuracy << ",";
    csv_file << data.miou << ",";
    csv_file << data.tp << ",";
    csv_file << data.tn << ",";
    csv_file << data.fp << ",";
    csv_file << data.fn << ",";
    csv_file << data.exec_time << ",";
    csv_file << data.ground_size << ",";
    csv_file << data.truss_size << ",";

    csv_file << data.density_enable << ",";
    csv_file << data.density_first << ",";
    csv_file << data.density_radius << ",";
    csv_file << data.density_threshold << ",";

    csv_file << data.euclid_enable << ",";
    csv_file << data.euclid_radius << ",";
    csv_file << data.euclid_min_size << "\n";

    csv_file.close();
}
#endif // CSV_UTILS_HPP