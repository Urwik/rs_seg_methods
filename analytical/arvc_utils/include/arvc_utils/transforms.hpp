#pragma once
#include <iostream>
#include <Eigen/Dense>

namespace arvc 
{

    Eigen::Affine3f applyTfNoise(const Eigen::Affine3f& tf, const float& _noise);

}