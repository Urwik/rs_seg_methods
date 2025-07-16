#include <filesystem>
#include <iostream>
#include <Eigen/Eigen>
// #include <gazebo/gazebo.hh>
#include <gazebo-11/gazebo/gazebo.hh>
#include <sdf/sdf.hh>
#include <ros/ros.h>
#include <ros/package.h>
#include <vector>
#include <tuple>
#include <sdf/Polyline.hh>
#include <sstream>

#include "arvc_sdf.hpp"
#include "arvc_console.hpp"
#include <yaml-cpp/yaml.h>

arvc::console cons;

namespace fs = std::filesystem;
using namespace std;


class trussBuilder{


  public:
  trussBuilder(const string& _name){

    this->laser_retro_count = 1;
    this->name = _name;

    this->bl_x.reset(new sdf::Element());
    this->bl_y.reset(new sdf::Element());
    this->bl_z.reset(new sdf::Element());
    this->bl_intersection.reset(new sdf::Element());

    this->basic_link.reset(new sdf::Element());
    this->textured = false;
  }

  trussBuilder(){}

  ~trussBuilder(){}


  void change_config_model_name(fs::path _model_path){

    fs::path config_path = _model_path / "model.config";

    std::ifstream file(config_path.string());

    if (!file) {
        std::cout << "Unable to open file";
        return;
    }

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) {
        // Change the line if it matches a certain condition
        if (line == "    <name>model_name</name>") {
            line = "    <name>" + this->name + "</name>";
        }
        lines.push_back(line);
    }

    file.close();

    // Write the changes back to the file
    std::ofstream outfile(config_path.string());
    for (const auto &line : lines) {
        outfile << line << "\n";
    }
  }


  void create_model_folder(){


    cons.debug("CREATING MODEL FOLDER");
    this->MODELS_PKG_PATH = ros::package::getPath("truss_generator") + "/models/";
    this->path = fs::path(this->MODELS_PKG_PATH / this->name);
    fs::copy(this->MODELS_PKG_PATH / "BASE_MODELS" / "TEMPLATE", this->path, fs::copy_options::recursive | fs::copy_options::overwrite_existing);

    // TODO: read the .config file and set the model name to this->name

    change_config_model_name(this->path);

    // cons.debug("Changing config. model name");
    // sdf::SDFPtr _tmp_sdf_ptr (new sdf::SDF());
    // sdf::init(_tmp_sdf_ptr);
    // sdf::readStringWithoutConversion(this->path.string() + "/model.config", _tmp_sdf_ptr);

    // sdf::ElementPtr modelElement = _tmp_sdf_ptr->Root()->GetElement("model");
    // modelElement->GetAttribute("name")->Set<string>(this->name);
    // _tmp_sdf_ptr->Write(this->path.string() + "/model.config");

    // cons.debug("Model folder created and config. updated");
  }


  void init_sdf_ptr(){

    cons.debug("INITIALIZING SDF PTR");
    this->sdf_ptr = sdf::SDFPtr(new sdf::SDF());
    sdf::init(this->sdf_ptr);
    sdf::readFile(this->path.string() + "/model.sdf", this->sdf_ptr);
  }


  void set_model_name(){
    cons.debug("SETTING MODEL NAME");
    this->model = this->sdf_ptr->Root()->GetElement("model");
    this->model->GetAttribute("name")->Set<string>(this->name);
  }


  void get_basic_links(){

    cons.debug("GETTING BASIC LINKS");
    sdf::SDFPtr _tmp_sdf_ptr (new sdf::SDF());
    sdf::init(_tmp_sdf_ptr);
    
    if(this->textured and !this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_textured/model.sdf", _tmp_sdf_ptr);
    else if (this->textured and this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_textured_unlabel/model.sdf", _tmp_sdf_ptr);
    else if (!this->textured and this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_unlabel/model.sdf", _tmp_sdf_ptr);
    else
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2/model.sdf", _tmp_sdf_ptr);

    this->basic_link = _tmp_sdf_ptr->Root()->GetElement("model");
    this->basic_link = this->basic_link->GetElement("link");


    this->bl_x->Copy(this->basic_link);
    this->bl_y->Copy(this->basic_link);
    this->bl_z->Copy(this->basic_link);
    this->bl_intersection->Copy(this->basic_link);

    if (this->shadowed) {
      this->bl_x->GetElement("visual")->GetElement("cast_shadows")->Set<bool>(true);
      this->bl_y->GetElement("visual")->GetElement("cast_shadows")->Set<bool>(true);
      this->bl_z->GetElement("visual")->GetElement("cast_shadows")->Set<bool>(true);
    }


    this->x_step = this->node_length + this->node_width;
    this->y_step = this->node_length + this->node_width;
    this->z_step = this->node_length + this->node_width;
    this->offset = (this->node_width + this->node_length) / 2;
  }


  void set_links_size() {
    cons.debug("SETTING LINKS SIZE");
    arvc::setScale(this->bl_x, im::Vector3d(this->node_length, this->node_width, this->node_width));
    arvc::setScale(this->bl_y, im::Vector3d(this->node_width, this->node_length, this->node_width));
    arvc::setScale(this->bl_z, im::Vector3d(this->node_width, this->node_width, this->node_length));
    arvc::setScale(this->bl_intersection, im::Vector3d(this->node_width, this->node_width, this->node_width));
  }


  void insert_link(sdf::ElementPtr _link){

    cons.debug("INSERTING LINK");
    sdf::ElementPtr copy_link = _link->Clone();
    sdf::ElementPtr visualElement = copy_link->GetElement("visual");


    // INCREASE LASER RETRO FOR EACH VISUAL ELEMENT
    while (visualElement) {
      sdf::ElementPtr retroElement = visualElement->GetElement("laser_retro");
      retroElement->Set<int>(this->laser_retro_count);
      visualElement = visualElement->GetNextElement("visual");
      this->laser_retro_count++;
    }
    

    this->model->InsertElement(copy_link);
  }


  void run(){
    this->create_model_folder();
    this->init_sdf_ptr();
    this->set_model_name();
    this->get_basic_links();
    this->set_links_size();
    this->build();
    this->save_model();
  }


  void build(){

    float x = 0, y = 0, z = 0;
    for (size_t k = 0; k < this->truss_size[0]; k++) // Z 
    {
      for (size_t j = 0; j < this->truss_size[1]+1; j++) // Y
      {
        for (size_t i = 0; i < this->truss_size[2]+1; i++) // X
        {
          x = i * this->x_step;
          y = j * this->y_step;
          z = k * this->z_step;

          // X LIMIT NOT REACHED
          if ( i < this->truss_size[2]){ 
            arvc::setName(this->bl_x, "x_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_x, im::Vector3d(x + this->offset, y, (z + this->offset)));
            this->insert_link(this->bl_x);
          }          

          // Y LIMIT NOT REACHED
          if( j < this->truss_size[1]){ 
            arvc::setName(this->bl_y, "y_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_y, im::Vector3d(x, y + this->offset, (z + this->offset)));
            this->insert_link(this->bl_y);
          }
          
          if (this->optimize){ 
            if ((k == 0 or k == this->truss_size[0] - 1) or (j == 0 or j == this->truss_size[1]) or (i == 0 or i == this->truss_size[2])) {
              arvc::setName(this->bl_intersection, "u_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
              arvc::setPosition(this->bl_intersection, im::Vector3d(x, y, (z + this->offset)));
              this->insert_link(this->bl_intersection);
            }
          }
          else {
            arvc::setName(this->bl_intersection, "u_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_intersection, im::Vector3d(x, y, (z + this->offset)));
            this->insert_link(this->bl_intersection);
          }

          if (!this->use_stands and k == 0)
            continue;
          else {
            arvc::setName(this->bl_z, "z_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_z, im::Vector3d(x, y, z));
            this->insert_link(this->bl_z);
          }
        }
      }
    }
  }


  void save_model(){
    cons.debug("SAVING MODEL");
    this->sdf_ptr->Write(this->path.string() + "/model.sdf");
  }


  protected:
  sdf::SDFPtr sdf_ptr;
  fs::path path, MODELS_PKG_PATH;
  int laser_retro_count;

  
  float x_step, y_step, z_step, offset;
  sdf::ElementPtr basic_link, bl_x, bl_y, bl_z, bl_intersection;
  
  public:
  string name;
  sdf::ElementPtr model;
  im::Vector3i truss_size;
  float node_length;
  float node_width;
  bool textured, unlabeled, shadowed;
  bool use_stands;
  bool optimize;

};


class trussCrossedBuilder : public trussBuilder{
  
  public:
  sdf::ElementPtr diagonal_link, dl_x, dl_y;
  im::Vector3d o_rot_dx, o_rot_dy;

  bool intercale_dir;


  trussCrossedBuilder(const string& _name) : trussBuilder(_name){
    this->diagonal_link.reset(new sdf::Element());
    this->dl_x.reset(new sdf::Element());
    this->dl_y.reset(new sdf::Element());

    this->intercale_dir = false;
  }

  void setup_base_unlabeled_parallelogram(){
    // COMPUTE PARALLELOGRAM PARAMETERS
    arvc::parallelogram p;
    p.length = this->node_length;
    p.height = this->node_length;
    p.width = this->node_width;
    p.compute_parameters();


    // SET PARALLELOGRAM PARAMETERS0
    sdf::ElementPtr visualElement (new sdf::Element());
    sdf::ElementPtr polylineElement (new sdf::Element());
    sdf::ElementPtr poseElement (new sdf::Element());
    sdf::ElementPtr pointElement (new sdf::Element());
    sdf::ElementPtr widthElement (new sdf::Element());
    im::Pose3d pose_2;
    sdf::Polyline polyline;


    visualElement = this->diagonal_link->GetElement("visual");

    const float half_length = this->node_length / 2;
    const float half_width = this->node_width / 2;

    cons.debug("Visual Element Name: " + visualElement->GetAttribute("name")->GetAsString());
    poseElement = visualElement->GetElement("pose");
    poseElement->Set<im::Pose3d>(im::Pose3d(0, 0, -half_width, 0, 0, 0));

    polylineElement = visualElement->GetElement("geometry")->GetElement("polyline");

    widthElement = polylineElement->GetElement("height");
    widthElement->Set<float>(this->node_width);

    pointElement = polylineElement->GetElement("point");
    cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    pointElement->Set<im::Vector2d>(im::Vector2d(- half_length, - half_length));
    cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    
    pointElement = pointElement->GetNextElement("point");
    cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    pointElement->Set<im::Vector2d>(im::Vector2d(-half_length + p.a, - half_length));
    cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    
    pointElement = pointElement->GetNextElement("point");
    cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    pointElement->Set<im::Vector2d>(im::Vector2d(half_length, half_length));
    cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    
    pointElement = pointElement->GetNextElement("point");
    cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
    pointElement->Set<im::Vector2d>(im::Vector2d(half_length - p.a, half_length));
    cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");


  }


  void setup_base_parallelogram(){
    
    // COMPUTE PARALLELOGRAM PARAMETERS
    arvc::parallelogram p;
    p.length = this->node_length;
    p.height = this->node_length;
    p.width = this->node_width;
    p.compute_parameters();


    // SET PARALLELOGRAM PARAMETERS0
    sdf::ElementPtr visualElement (new sdf::Element());
    sdf::ElementPtr polylineElement (new sdf::Element());
    sdf::ElementPtr poseElement (new sdf::Element());
    sdf::ElementPtr pointElement (new sdf::Element());
    sdf::ElementPtr boxSizeElement (new sdf::Element());
    im::Pose3d pose_2;
    sdf::Polyline polyline;


    visualElement = this->diagonal_link->GetElement("visual");

    const float half_length = this->node_length / 2;
    const float half_width = this->node_width / 2;
    
    for (size_t i = 0; i < 4; i++)
    {
      cons.debug("Visual Element Name: " + visualElement->GetAttribute("name")->GetAsString());
      poseElement = visualElement->GetElement("pose");

      switch (i)
      {
      case 0:
        polylineElement = visualElement->GetElement("geometry")->GetElement("polyline");

/*         // USE OF sdf::Polyline
        // polyline.ClearPoints();
        // polyline.Load(polylineElement);

        // polyline.PointByIndex(0)->Set(- half_length, - half_length);
        // polyline.PointByIndex(1)->Set(-half_length + p.a, - half_length);
        // polyline.PointByIndex(2)->Set(half_length, half_length);
        // polyline.PointByIndex(3)->Set(half_length - p.a, half_length);
        // polylineElement->GetElement("point")->Set<im::Vector2d>(im::Vector2d(- half_length, - half_length));
        // polylineElement->Set<sdf::Polyline>(polyline); */

        pointElement = polylineElement->GetElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(- half_length, - half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(-half_length + p.a, - half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
       
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(half_length, half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
       
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(half_length - p.a, half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");

        poseElement->Set<im::Pose3d>(im::Pose3d(0, 0, half_width, 0, 0, 0));
        break;
      
      case 1:
        polylineElement = visualElement->GetElement("geometry")->GetElement("polyline");

/*         // USE OF sdf::Polyline
        // polyline.ClearPoints();
        // polyline.Load(polylineElement);

        // polyline.PointByIndex(0)->Set(- half_length, - half_length);
        // cons.debug("Point 0: " + to_string(polyline.PointByIndex(0)->X()) + ", " + to_string(polyline.PointByIndex(0)->Y()));

        // polyline.PointByIndex(1)->Set(-half_length + p.a, - half_length);
        // cons.debug("Point 1: " + to_string(polyline.PointByIndex(1)->X()) + ", " + to_string(polyline.PointByIndex(1)->Y()));

        // polyline.PointByIndex(2)->Set(half_length, half_length);
        // cons.debug("Point 2: " + to_string(polyline.PointByIndex(2)->X()) + ", " + to_string(polyline.PointByIndex(2)->Y()));

        // polyline.PointByIndex(3)->Set(half_length - p.a, half_length);
        // cons.debug("Point 3: " + to_string(polyline.PointByIndex(3)->X()) + ", " + to_string(polyline.PointByIndex(3)->Y()));
        // polylineElement->Set<sdf::Polyline>(polyline); */


        pointElement = polylineElement->GetElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(- half_length, - half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");

        
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(-half_length + p.a, - half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");

       
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(half_length, half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");

       
        pointElement = pointElement->GetNextElement("point");
        cons.debug("Original Point: [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");
        pointElement->Set<im::Vector2d>(im::Vector2d(half_length - p.a, half_length));
        cons.debug("New Point     : [" + to_string(pointElement->Get<im::Vector2d>().X()) + ", " + to_string(pointElement->Get<im::Vector2d>().Y()) + "]");


        poseElement->Set<im::Pose3d>(im::Pose3d(0, 0, -half_width, 0, 0, 0));
        break;

      case 2:
        boxSizeElement = visualElement->GetElement("geometry")->GetElement("box")->GetElement("size");
        cons.debug("Original Box: " + boxSizeElement->ToString(" "));
        boxSizeElement->Set<im::Vector3d>(im::Vector3d(p.b, 0.000001, this->node_width));
        cons.debug("New Box     : " + boxSizeElement->ToString(" "));

        cons.debug("Original Pose: " + poseElement->ToString(" "));
        pose_2.Set(this->node_width / (2 * sin(p.alpha)), 0, 0, 0, 0, p.alpha);
        poseElement->Set<im::Pose3d>(pose_2);
        cons.debug("New Pose     : " + poseElement->ToString(" "));
        break;

      case 3:
        boxSizeElement = visualElement->GetElement("geometry")->GetElement("box")->GetElement("size");
        cons.debug("Original Box: " + boxSizeElement->ToString(" "));
        boxSizeElement->Set<im::Vector3d>(im::Vector3d(p.b, 0.000001, this->node_width));
        cons.debug("New Box     : " + boxSizeElement->ToString(" "));

        cons.debug("Original Pose: " + poseElement->ToString(" "));
        pose_2.Set(-this->node_width / (2 * sin(p.alpha)), 0, 0, 0, 0, p.alpha); 
        poseElement->Set<im::Pose3d>(pose_2);
        cons.debug("New Pose     : " + poseElement->ToString(" "));
        break;

      default:
        break;
      }

      if(i < 3)
        visualElement = visualElement->GetNextElement("visual");
    }
  }


  void get_diagonal_link(){
    cons.debug("GETTING DIAGONAL LINK");

    sdf::SDFPtr _tmp_sdf_ptr (new sdf::SDF());
    sdf::init(_tmp_sdf_ptr);
    sdf::ElementPtr original_link;

    // if(this->textured and !this->unlabeled)
    //   sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_textured/model.sdf", _tmp_sdf_ptr);
    // else if (this->textured and this->unlabeled)
    //   sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_textured_unlabel/model.sdf", _tmp_sdf_ptr);
    // else if (!this->textured and this->unlabeled)
    //   sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2_unlabel/model.sdf", _tmp_sdf_ptr);
    // else
    //   sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_block_v2/model.sdf", _tmp_sdf_ptr);



    // GET A COPY OF THE LINK OF THE BASE MODEL
    if(this->textured and !this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_parallelogram_textured/model.sdf", _tmp_sdf_ptr);
    else if (this->textured and this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_parallelogram_textured_unlabel/model.sdf", _tmp_sdf_ptr);
    else if (!this->textured and this->unlabeled)
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_parallelogram_unlabel/model.sdf", _tmp_sdf_ptr);
    else 
      sdf::readFile(this->MODELS_PKG_PATH / "BASE_MODELS" / "simple_parallelogram/model.sdf", _tmp_sdf_ptr);

    cons.debug("SIMPLE PARALLELOGRAM LINK READ");

    original_link = _tmp_sdf_ptr->Root()->GetElement("model");
    original_link = original_link->GetElement("link");
    this->diagonal_link->Copy(original_link);
    cons.info("DIAGONAL LINK COPIED");

    // SET THE DIAGONAL LINK PARAMETERS (ADJUST TO LENGHT OF THE TRUSS NODES)
    if (!this->unlabeled)
      this->setup_base_parallelogram();
    else
      this->setup_base_unlabeled_parallelogram();

    cons.info("DIAGONAL LINK PARAMETERS SET");

    // SET SPAWNING LINKS IN X AND Y DIRECTIONS
    this->dl_x->Copy(this->diagonal_link);
    this->dl_y->Copy(this->diagonal_link);

    if (this->shadowed) {
      this->dl_x->GetElement("visual")->GetElement("cast_shadows")->Set<bool>(true);
      this->dl_y->GetElement("visual")->GetElement("cast_shadows")->Set<bool>(true);
    }

    this->o_rot_dx = im::Vector3d(M_PI/2, 0, 0);
    this->o_rot_dy = im::Vector3d(M_PI/2, 0, M_PI/2);

    arvc::setRotation(this->dl_x, this->o_rot_dx);
    arvc::setRotation(this->dl_y, this->o_rot_dy);
  }


  void build(){

    float x = 0, y = 0, z = 0;
    for (size_t k = 0; k < this->truss_size[0]; k++)
    {
      for (size_t j = 0; j < this->truss_size[1]+1; j++)
      {
        for (size_t i = 0; i < this->truss_size[2]+1; i++)
        {
          x = i * this->x_step;
          y = j * this->y_step;
          z = k * this->z_step;

          if ( i < this->truss_size[2]){
            arvc::setName(this->bl_x, "x_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_x, im::Vector3d(x + this->offset, y, (z + this->offset)));
            this->insert_link(this->bl_x);

            if  (k > 0){
              if(this->intercale_dir){
                if(j % 2 == 0)
                  arvc::setRotation(this->dl_x, this->o_rot_dx);
                else
                  arvc::setRotation(this->dl_x, im::Vector3d(M_PI/2, 0, M_PI));
              }

              arvc::setName(this->dl_x, "dx_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
              arvc::setPosition(this->dl_x, im::Vector3d(x + this->offset, y, z));
              cons.debug("DX LINK VISUAL 0: " + this->dl_x->GetElement("visual")->GetElement("geometry")->GetElement("polyline")->GetElement("point")->ToString(" "));

              this->insert_link(this->dl_x);
            }
          }          
          
          if( j < this->truss_size[1]){
            arvc::setName(this->bl_y, "y_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_y, im::Vector3d(x, y + this->offset, (z + this->offset)));
            this->insert_link(this->bl_y);
            if  (k > 0){
              if(this->intercale_dir){
                if(i % 2 == 0)
                  arvc::setRotation(this->dl_y, this->o_rot_dy);
                else
                  arvc::setRotation(this->dl_y, im::Vector3d(M_PI/2, 0, 1.5*M_PI));
              }
              
              arvc::setName(this->dl_y, "dy_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
              arvc::setPosition(this->dl_y, im::Vector3d(x, y + this->offset, z));
              cons.debug("DY LINK VISUAL 0: " + this->dl_y->GetElement("visual")->GetElement("geometry")->GetElement("polyline")->GetElement("point")->ToString(" "));


              this->insert_link(this->dl_y);
            }
          }


          if (this->optimize){ 
            if ((k == 0 or k == this->truss_size[0] - 1) or (j == 0 or j == this->truss_size[1]) or (i == 0 or i == this->truss_size[2])) {
              arvc::setName(this->bl_intersection, "u_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
              arvc::setPosition(this->bl_intersection, im::Vector3d(x, y, (z + this->offset)));
              this->insert_link(this->bl_intersection);
            }
          }
          else {
            arvc::setName(this->bl_intersection, "u_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_intersection, im::Vector3d(x, y, (z + this->offset)));
            this->insert_link(this->bl_intersection);
          }

          if (!this->use_stands and k == 0)
            continue;
          else {
            arvc::setName(this->bl_z, "z_link_" + std::to_string(k) + "_" + std::to_string(j) + "_" + std::to_string(i));
            arvc::setPosition(this->bl_z, im::Vector3d(x, y, z));
            this->insert_link(this->bl_z);
          }
        }
      }
    }
    this->save_model();
  }


  void run(){

    cons.info("RUNNING TRUSS CROSSED BUILDER");
    cout << "NAME: " << this->name << endl;
    this->create_model_folder();
    this->init_sdf_ptr();
    this->set_model_name();
    this->get_basic_links();
    this->set_links_size();
    this->get_diagonal_link();
    this->build();
    this->save_model();
  }

};



int main(int argc, char const *argv[])
{
  std::cout << "Generating truss model..." << std::endl;

  YAML::Node config = YAML::LoadFile(ros::package::getPath("truss_generator") + "/config/config.yaml");

  int dim_x = config["dim"]["X"].as<int>();
  int dim_y = config["dim"]["Y"].as<int>();
  int dim_z = config["dim"]["Z"].as<int>();

  std::string model_name  = config["model_name"].as<std::string>();
  std::string model_type  = config["type"].as<std::string>();
  float node_length       = config["node"]["length"].as<float>();
  float node_width        = config["node"]["width"].as<float>();
  bool textured           = config["textured"].as<bool>();
  bool intercale_dir      = config["intercale_dir"].as<bool>();
  bool unlabeled          = config["unlabeled"].as<bool>();
  bool shadowed           = config["shadowed"].as<bool>();
  bool use_stands         = config["use_stands"].as<bool>();
  bool optimize           = config["optimize"].as<bool>();


  if (model_name == "") {

    std::stringstream stream;
    stream << std::fixed << std::setprecision(2) << node_length << node_width;
    std::string s = stream.str();

    // model_name = "truss_" + model_type + "_" + std::to_string(dim_x) + std::to_string(dim_y) + std::to_string(dim_z) + "_" + s;

    model_name = model_type + "_" + std::to_string(dim_x) + std::to_string(dim_y) + std::to_string(dim_z) + "_" + s + "_";

    if (textured)
      model_name += "T";
    if (intercale_dir and model_type == "crossed")
      model_name += "I";
    if (unlabeled)
      model_name += "U";
    if (shadowed)
      model_name += "S";
  }

  cons.enable = false;
  cons.enable_debug = false;

  if (config["type"].as<std::string>() == "crossed")
  {
    trussCrossedBuilder tcb(model_name);
    tcb.truss_size = im::Vector3i(dim_z, dim_y, dim_x); // Z Y X
    tcb.unlabeled = unlabeled;
    tcb.node_length = node_length;
    tcb.node_width = node_width;
    tcb.textured = textured;
    tcb.intercale_dir = intercale_dir;
    tcb.shadowed = shadowed;
    tcb.use_stands = use_stands;
    tcb.optimize = optimize;
    tcb.run();
  }
  else if (config["type"].as<std::string>() == "orthogonal")
  {
    trussBuilder tb(model_name);
    tb.truss_size = im::Vector3i(dim_z, dim_y, dim_x); // Z Y X
    tb.unlabeled = unlabeled;
    tb.node_length = node_length;
    tb.node_width = node_width;
    tb.textured = textured;
    tb.shadowed = shadowed;
    tb.use_stands = use_stands;
    tb.optimize = optimize;
    tb.run();
  }
  else
  {
    cons.error("Invalid truss type");
  }
  

  std::cout << GREEN << "Truss model generated!" << RESET << std::endl;
  return 0;
}

