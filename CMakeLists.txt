cmake_minimum_required(VERSION 3.13)
project(object_detection_and_classification)

set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)

include_directories(src)

add_executable(object_detection_and_classification
        src/config.h
        src/hog_visualization.cpp
        src/hog_visualization.h
        src/main.cpp
        src/ObjectDetector.h
        src/progress.h
        src/RandomForest.cpp
        src/RandomForest.hpp
        src/task1.h
        src/utils.h)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_detection_and_classification ${OpenCV_LIBS} )

