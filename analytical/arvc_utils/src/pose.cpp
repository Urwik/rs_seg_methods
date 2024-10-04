#include "arvc_utils/pose.hpp"

std::vector<Eigen::Affine3f> arvc::loadPosesVector(const fs::path& file_path)
{
    std::cout << "Loading poses from: " << file_path << std::endl;

    std::vector<Eigen::Affine3f> poses;
    YAML::Node poses_yaml = YAML::LoadFile(file_path.string());

    for(YAML::const_iterator it = poses_yaml.begin() ; it != poses_yaml.end() ; ++it) {

        std::string node_name = it->first.as<std::string>();
        
        Eigen::Affine3f pose;

        pose.translation().x() = poses_yaml[node_name]["translation"]["x"].as<float>();
        pose.translation().y() = poses_yaml[node_name]["translation"]["y"].as<float>();
        pose.translation().z() = poses_yaml[node_name]["translation"]["z"].as<float>();
 
        Eigen::Quaternionf q;   
        q.x() = poses_yaml[node_name]["rotation"]["x"].as<float>();
        q.y() = poses_yaml[node_name]["rotation"]["y"].as<float>();
        q.z() = poses_yaml[node_name]["rotation"]["z"].as<float>();
        q.w() = poses_yaml[node_name]["rotation"]["w"].as<float>();

        pose.linear() = q.toRotationMatrix();        

        poses.push_back(pose);
    }
    return poses;
}