cmake_minimum_required(VERSION 3.0.2)
project(analytical_rs_seg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

## Expose this project's source directory (where project() is called)
## Use PROJECT_SOURCE_DIR to avoid pointing to the top-level superproject
add_definitions(-DPROJECT_PATH=\"${PROJECT_SOURCE_DIR}\")


## For compilation with ROS Noetic uncomment catkin

set(ROS_BUILD_ENABLE TRUE)


if (ROS_BUILD_ENABLE)
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    roslib
  )
endif()


find_package(PCL 1.10 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Yaml-cpp REQUIRED)
find_package(Threads REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS} ${Eigen3_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS} ${Eigen3_LIBRARY_DIRS} ${YAML_CPP_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS} ${Eigen3_DEFINITIONS} ${YAML_CPP_DEFINITIONS})


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need

if (ROS_BUILD_ENABLE)
  catkin_package(
    INCLUDE_DIRS include
    LIBRARIES rs_ground_filter
    CATKIN_DEPENDS roscpp rospy roslib
  )
endif()


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )



add_library(rs_ground_filter src/ground_filter.cpp)

if (ROS_BUILD_ENABLE)
  target_link_libraries(rs_ground_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
else()
  target_link_libraries(rs_ground_filter ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
endif()



add_executable(ground_filter_node src/node.cpp)
if (ROS_BUILD_ENABLE)
  target_link_libraries(ground_filter_node rs_ground_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
else()
  target_link_libraries(ground_filter_node rs_ground_filter ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
endif()

if (ROS_BUILD_ENABLE)
  add_executable(run_experiment_queue src/experiment_queue_v2.cpp)
  target_link_libraries(run_experiment_queue rs_ground_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
endif()

if (ROS_BUILD_ENABLE)
  add_executable(find_methods_difference src/find_method_differences_in_clouds.cpp)
  target_link_libraries(find_methods_difference rs_ground_filter ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${Eigen3_LIBRARIES} ${YAML_CPP_LIBRARIES})
endif()