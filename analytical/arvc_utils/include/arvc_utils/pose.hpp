#pragma once

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <yaml-cpp/yaml.h>
#include <filesystem>

namespace fs = std::filesystem;

namespace arvc {

std::vector<Eigen::Affine3f> loadPosesVector(const fs::path& file_path);

};