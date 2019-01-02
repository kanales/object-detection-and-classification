//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "hog_visualization.h"
#include "config.h"
#include "utils.h"
#include "task1.h"
#include "random_forest.h"

cv::Mat create_test(cv::String test_path)
{
    std::vector<std::string> v;
    // test data
    cv::String path2(test_path + "01/");
    read_directory(path2, v);
    cv::Mat testData = cv::Mat((int)v.size(), 979104, CV_32F);
    int iter = 0;
    for (auto &i : v) {
        cv::Mat m = cv::Mat(task1(i)).t();
        // m.convertTo( m, CV_32F );
        m.copyTo(testData.row(iter));
        iter++;
    }
    return testData;
}

void print(const int& n) {
    std::cout << n << " ";
}

int main(int argc, const char * argv[]) {
    
    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    cv::String path( $ROOT "data/task2/train/0" );
    cv::String path2( $ROOT "data/task2/test/" );
    
    int nsample = 50;
    
    if( argc > 1)
    {
        imageName = argv[1];
        // string imageName = "./data/task1/obj1000.jpg";
    }
    
    // TASK2
    
    random_forest rf(10,nsample, 6);
    
    std::cout << "Training forest..." << std::endl;
    rf.train(path);
    std::cout << "Done training." << std::endl;
    
    std::cout << "Predicting..." << std::endl;
    cv::Mat test  = create_test(path2);
    
    for (int i = 0; i < test.rows; i++) {
        int pred = rf.predict(test);
        std::cout << pred << std::endl;
    }
    
    return 0;
}
