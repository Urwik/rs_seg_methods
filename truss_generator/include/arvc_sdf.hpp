#pragma once

#include <sdf/sdf.hh>
#include <ignition/math.hh>
#include <filesystem>

namespace im = ignition::math;
namespace fs = std::filesystem;
using namespace std;
namespace arvc{

  std::tuple<sdf::ElementPtr, sdf::SDFPtr> getRootElement( fs::path _sdf_path){

    sdf::SDFPtr sdf_File (new sdf::SDF());
    sdf::init(sdf_File);
    
    sdf::readFile(_sdf_path, sdf_File);

    sdf::ElementPtr modelElement (new sdf::Element());
    modelElement = sdf_File->Root()->GetElement("model");

    return  std::tuple(modelElement, sdf_File);
}


  void setScale(sdf::ElementPtr _model, const im::Vector3d& _scale) {

    // std::cout << "Setting Scale: " << std::endl;

    // sdf::ElementPtr visualElement = _model->GetElement("link")->GetElement("visual");
    sdf::ElementPtr visualElement = _model->GetElement("visual");


    while (visualElement)
    {
      // cout << "Modifying Visual Element: " << visualElement->GetAttribute("name")->GetAsString() << endl;
      sdf::ElementPtr sizeElement = visualElement->GetElement("geometry")->GetElement("box")->GetElement("size");
      sdf::ElementPtr poseElement = visualElement->GetElement("pose");

      im::Vector3d size =  sizeElement->Get<im::Vector3d>();

      for (size_t i = 0; i < 3; i++)
      {
        if (size[i] > 0.01)
          size[i] = size[i] * _scale[i];
      }
      
      im::Pose3d pose =  poseElement->Get<im::Pose3d>();
      pose.Pos() = pose.Pos() * _scale;

      sizeElement->Set<im::Vector3d>(size);
      poseElement->Set<im::Pose3d>(pose);

      visualElement = visualElement->GetNextElement("visual");
    }

    // Set Scale to Collision element 
    // sdf::ElementPtr collisionElement = _model->GetElement("link")->GetElement("collision");
    sdf::ElementPtr collisionElement = _model->GetElement("collision");
    sdf::ElementPtr sizeElement = collisionElement->GetElement("geometry")->GetElement("box")->GetElement("size");
    sizeElement->Set<im::Vector3d>(_scale);
  }


  void setName(sdf::ElementPtr _link, const std::string& _name){
    _link->GetAttribute("name")->SetFromString(_name);
  }


  void setPosition(sdf::ElementPtr _link, const im::Vector3d& _position){
    sdf::ElementPtr poseElement = _link->GetElement("pose");
    im::Pose3d pose = poseElement->Get<im::Pose3d>();

    im::Pose3d new_pose;
    new_pose.Set(_position, pose.Rot().Euler());

    poseElement->Set<im::Pose3d>(new_pose);
  }


  void setRotation(sdf::ElementPtr _link, const im::Vector3d & _rotation){
    sdf::ElementPtr poseElement = _link->GetElement("pose");
    im::Pose3d pose = poseElement->Get<im::Pose3d>();

    im::Pose3d new_pose;
    new_pose.Set(pose.Pos(), _rotation);

    poseElement->Set<im::Pose3d>(new_pose);
  }


  void setPose(sdf::ElementPtr _link, const im::Pose3d& _pose){
    setPosition(_link, _pose.Pos());
    setRotation(_link, _pose.Rot().Euler());
  }

class parallelogram
{
  private:

  public:
    parallelogram(/* args */){}
    ~parallelogram(){}

  void compute_parameters(){
    this->betha = atan2(this->height, this->length);
    this->c = sqrt(pow(this->height, 2) + pow(this->length, 2));
    float _a, _b, _c;

    // _ax² + _bx + _c = 0
    _a = 1 - (pow(this->width, 2) / pow(this->height, 2));
    _b = (2 * this->width * this->c * cos(this->betha)) / this->height;
    _c = -(pow(this->length, 2) + pow(this->height, 2));

    float b1 = (-_b + sqrt((_b * _b) - (4 * _a * _c))) / (2 * _a);
    float b2 = (-_b - sqrt((_b * _b) - (4 * _a * _c))) / (2 * _a);

    if (b1 > 0) this->b = b1;  else this->b = b2;

    this->a = b * (this->width / this->height);
    this->alpha = asin(this->width / this->a);

    cout << "Parallelogram: " << endl;
    cout << "\ta: " << this->a << endl;
    cout << "\tb: " << this->b << endl;
    cout << "\tc: " << this->c << endl;
    cout << "\tbetha: " << this->betha << endl;
    cout << "\talpha: " << this->alpha << endl;

  }
  
  float length; // Length of the base (x)
  float height; // Height of the parallelogram (y)
  float width;  // Width of the parallelogram (z)
  float betha, c;
  float a, b, alpha; // Sides and angle of the parallelogram  
};
  

} // namespace arvc