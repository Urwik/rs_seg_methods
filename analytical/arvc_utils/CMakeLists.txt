cmake_minimum_required(VERSION 3.0.2)
project(arvc_utils VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(catkin REQUIRED COMPONENTS roscpp)

find_package(PCL 1.10 REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(gazebo REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(VTK REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  # ${VTK_INCLUDE_DIRS}
  )

link_directories(
  ${catkin_LIBRARY_DIRS} 
  ${PCL_LIBRARY_DIRS} 
  ${Eigen3_LIBRARY_DIRS} 
  ${GAZEBO_LIBRARY_DIRS}
  ${OpenCV_LIBRARY_DIRS}
  # ${VTK_LIBRARY_DIRS}
  )

add_definitions(
  ${catkin_DEFINITIONS} 
  ${PCL_DEFINITIONS} 
  ${Eigen3_DEFINITIONS} 
  ${GAZEBO_DEFINITIONS}
  ${OpenCV_DEFINITIONS}
  # ${VTK_DEFINITIONS}
  )


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
 INCLUDE_DIRS include
 CATKIN_DEPENDS roscpp
 LIBRARIES arvc_utils
#  DEPENDS system_lib
)

###########
## Build ##
###########
set(MY_LIB_FILES
    src/color.cpp 
    src/axes3d.cpp 
    src/console.cpp
    src/metrics.cpp 
    src/timer.cpp
    src/viewer.cpp 
    src/plane.cpp 
    src/transforms.cpp
    src/pose.cpp
    # "src/utils.cpp"
    )


add_library(arvc_utils ${MY_LIB_FILES})
# set_target_properties(arvc_utils PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(arvc_utils 
                      ${catkin_LIBRARIES} 
                      ${PCL_LIBRARIES} 
                      ${Eigen3_LIBRARIES} 
                      ${GAZEBO_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      # ${VTK_LIBRARIES}
                      )


#############
## Install ##
#############


## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

# ## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN ".svn" EXCLUDE
)