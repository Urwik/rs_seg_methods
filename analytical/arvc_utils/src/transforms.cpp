#include "arvc_utils/transforms.hpp"

namespace arvc 
{

Eigen::Affine3f applyTfNoise(const Eigen::Affine3f& tf, const float& _noise)
{
    Eigen::Affine3f noise_tf = tf;
    noise_tf.translation() += Eigen::Vector3f::Random() * _noise;
    
    noise_tf.rotate(Eigen::AngleAxisf(_noise, Eigen::Vector3f::UnitX()));
    noise_tf.rotate(Eigen::AngleAxisf(_noise, Eigen::Vector3f::UnitY()));
    noise_tf.rotate(Eigen::AngleAxisf(_noise, Eigen::Vector3f::UnitZ()));
    
    return noise_tf;
}

} // namespace arvc
