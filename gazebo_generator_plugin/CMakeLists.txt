cmake_minimum_required(VERSION 3.0.2)
project(gazebo_generator_plugin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# Setting compile flags based on build type
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")  # Debug information, no optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3")  # Optimized code for Release mode

# Apply the flags depending on build type
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  gazebo_ros
  pcl_conversions
  pcl_ros
  tf2_ros
)

find_package(Boost REQUIRED)
find_package(gazebo REQUIRED)
find_package(PCL 1.10 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Threads REQUIRED)
find_package(sdformat9 REQUIRED)


link_directories(${catkin_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
link_directories(${YAML_CPP_INCLUDE_DIRS})
link_directories(${Eigen_INCLUDE_DIRS})
link_directories(${SDFormat_LIBRARY_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
add_definitions(${catkin_DEFINITIONS})
add_definitions(${GAZEBO_DEFINITIONS})
add_definitions(${SDFormat_DEFINITIONS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(${YAML_CPP_DEFINITIONS})
add_definitions(${Boost_DEFINITIONS})



###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES arvc_gazebo_plugins
#  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include/generator_plugin
  ${SDFormat_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(arvc_gazebo_train_generator_plugin src/gazebo_train_generator_plugin.cpp)
target_link_libraries(arvc_gazebo_train_generator_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${Eigen_LIBRARIES} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} ${Boost_LIBRARIES}) 

add_library(arvc_gazebo_test_generator_plugin src/gazebo_test_generator_plugin.cpp)
target_link_libraries(arvc_gazebo_test_generator_plugin ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${Eigen_LIBRARIES} ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES} ${Boost_LIBRARIES}) 
