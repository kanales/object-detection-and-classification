cmake_minimum_required(VERSION 3.13)
project(object_detection_and_classification)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

include_directories(src)

add_executable(object_detection_and_classification
        src/config.h
        src/hog_visualization.cpp
        src/hog_visualization.h
        src/main.cpp
        src/RandomForest.cpp
        src/RandomForest.hpp
        src/task1.h
        src/utils.h src/ObjectDetector.cpp src/ObjectDetector.h src/task1.cpp src/utils.cpp src/task2.cpp src/task3.cpp src/RandomForest.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(object_detection_and_classification ${OpenCV_LIBS} )

