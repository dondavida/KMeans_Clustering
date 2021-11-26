cmake_minimum_required(VERSION 3.10)
project(image)

set (CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(src)

include_directories(${PROJECT_SOURCE_DIR}/src)

#find_package(OpenCV CONFIG REQUIRED)

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

#set(OpenCV_INSTALL_DIR "C:/MinGW/opencv2")
#Set(OpenCV_INCLUDE_DIRS "${OpenCV_INSTALL_DIR}/include/opencv;${OpenCV_INSTALL_DIR}/include/opencv2")
#set(OpenCV_INCLUDE_LIBS "${OpenCV_INSTALL_DIR}/lib")

#set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_gpu)


#include_directories(${OpenCV_INCLUDE_DIRS})
#add_executable(opencv opencv.cpp)

#target_link_libraries(opencv ${OpenCV_LIBS})