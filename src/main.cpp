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

#include "config.h"
#include "utils.h"
#include "task1.h"
#include "RandomForest.hpp"
#include "ObjectDetector.h"

std::vector<cv::Mat> load_test(cv::String test_path, char val) {
    cv::Mat image;
    std::vector<std::string> v;
    cv::String path2(test_path + "0" + val + "/");
    read_directory(path2, v);
    std::vector<cv::Mat> out(v.size());
    int idx = 0;
    for (auto &s: v) {
        out[idx++] = cv::imread(s, cv::IMREAD_COLOR);
    }
    return out;

//    std::vector<std::string> v;
//    // test data
//    cv::String path2(test_path + "0" + val + "/");
//    read_directory(path2, v);
//    cv::Mat testData = cv::Mat((int)v.size(), (int)hog.getDescriptorSize(), CV_32F);
//    int iter = 0;
//    for (auto &i : v) {
//        cv::Mat m = cv::Mat(extract_descriptor(hog,i)).t();
//        // m.convertTo( m, CV_32F );
//        m.copyTo(testData.row(iter));
//        iter++;
//    }
//    return testData;
}

template <class T>
void print_vector(std::vector<T> v) {
    std::cout << '[';
    for (T x: v) {
        std::cout << '\t' << x;
    }
    std::cout << " ]";
}

void part2(int argc, const char *argv[]) {

    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    cv::String path( $ROOT "data/task2/train/0" );
    cv::String path2( $ROOT "data/task2/test/" );

    int ntrees  = 30;
    int nsample = 150;

    if(argc > 1)
    {
        imageName = argv[1];
        // string imageName = "./data/task1/obj1000.jpg";
    }
    // TASK2
    cv::HOGDescriptor hog = mk_hog();
    RandomForest rf(ntrees,nsample, hog, 6);

    std::cout << "Training forest..." << std::endl;
    rf.train(path);
    std::cout << "Done training." << std::endl;

    std::cout << "Predicting..." << std::endl;

    char values[6] = {'0','1','2','3','4','5'};

    for (Class j = 0; j < 6; j++) {
        std::cout << "Expected: " << values[j] << ": " << std::endl;
        std::vector<cv::Mat> images = load_test(path2, values[j]);
        for (auto img: images) {
            std::vector<float> pred = rf.predictImage(img);
            int k = (int)std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
            std::cout << "\tP: " << k << ' ';
            print_vector(pred);
            std::cout << std::endl;
        }
    }
}

struct S {
    int a;
    int b;
};

int main(int argc, const char * argv[]) {
    part2(argc, argv);
//    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
//    cv::String path( $ROOT "data/task2/train/0" );
//    cv::String path2( $ROOT "data/task2/test/" );
//
//    int ntrees  = 25;
//    int nsample = 200;
//    Class bgClass = 0; // placeholder
//
//    cv::HOGDescriptor hog = mk_hog();
//    RandomForest rf(ntrees,nsample, hog, 6);
//    rf.train(path);
//    std::cout << "Done training." << std::endl;
//
//    ObjectDetector obd(rf, bgClass, 0);
//
//    obd.
}
