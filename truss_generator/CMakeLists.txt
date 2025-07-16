cmake_minimum_required(VERSION 3.0.2)
project(truss_generator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  roslib
  sensor_msgs
  std_msgs
)

find_package(gazebo REQUIRED)
find_package(sdformat9 EXACT REQUIRED)
find_package(yaml-cpp REQUIRED)

link_directories(${catkin_LIBRARY_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
link_directories(${SDFormat_LIBRARY_DIRS})
link_directories(${YAML_CPP_LIBRARY_DIRS})

add_definitions(${catkin_DEFINITIONS})
add_definitions(${GAZEBO_DEFINITIONS})
add_definitions(${SDFormat_DEFINITIONS})
add_definitions(${YAML_CPP_DEFINITIONS})

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


catkin_package(
#  INCLUDE_DIRS include
#  CATKIN_DEPENDS roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${SDFormat_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
)

add_executable(arvc_build_truss src/build_truss.cpp)
target_link_libraries(arvc_build_truss ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES} ${SDFormat_LIBRARIES} ${YAML_CPP_LIBRARIES}) 