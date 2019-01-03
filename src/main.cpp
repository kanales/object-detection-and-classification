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

cv::Mat create_test(cv::HOGDescriptor hog, cv::String test_path, char val)
{
    std::vector<std::string> v;
    // test data
    cv::String path2(test_path + "0" + val + "/");
    read_directory(path2, v);
    cv::Mat testData = cv::Mat((int)v.size(), (int)hog.getDescriptorSize(), CV_32F);
    int iter = 0;
    for (auto &i : v) {
        cv::Mat m = cv::Mat(extract_descriptor(hog,i)).t();
        // m.convertTo( m, CV_32F );
        m.copyTo(testData.row(iter));
        iter++;
    }
    return testData;
}

template <class T>
void print_vector(std::vector<T> v) {
    std::cout << '[';
    for (T x: v) {
        std::cout << ' ' << x;
    }
    std::cout << " ]";
}

void part2(int argc, const char *argv[]) {

    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    cv::String path( $ROOT "data/task2/train/0" );
    cv::String path2( $ROOT "data/task2/test/" );

    int ntrees  = 25;
    int nsample = 200;

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
    cv::Mat test;

    char values[6] = {'0','1','2','3','4','5'};

    for (int j=0; j < 6; j++) {
        std::cout << "Expected: " << values[j] << ": " << std::endl;
        test = create_test(hog, path2, values[j]);
        for (int i = 0; i < test.rows; i++) {
            std::vector<float> pred = rf.predict(test);
            int k = (int)std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
            std::cout << "\tP: " << k << ' ';
            print_vector(pred);
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

struct S {
    int a;
    int b;
};

int main(int argc, const char * argv[]) {
    //part2(argc, argv);
    std::vector<S> vec;

    for (int i = 0; i < 10; i++) {
        S s {i , -i};
        vec.push_back(s);
    }

    for (S x: vec) {
        std::cout << x.a << ' ' << x.b << std::endl;
    }
    return 0;
}
