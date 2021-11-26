find_package(OPENCV REQUIRED)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_calib3d opencv_video opencv_features2d opencv_ml opencv_highgui opencv_objdetect opencv_contrib opencv_legacy opencv_gpu)

include_directories(${OPENCV_INCLUDE_DIRS})


add_library(kmeans kmeans.cpp)

add_executable(main main.cpp)

target_link_libraries(main kmeans)

target_link_libraries(main ${OpenCV_LIBS})

