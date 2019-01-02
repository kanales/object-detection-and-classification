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

int main(int argc, const char * argv[]) {

    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    cv::String path( $ROOT "data/task2/train/0" );
    cv::String path2( $ROOT "data/task2/test/01/" );

    int nsample = 100;

    if( argc > 1)
    {
        imageName = argv[1];
        // string imageName = "./data/task1/obj1000.jpg";
    }

    // TASK2

    random_forest rf(10,nsample,0,6,10,16);
    rf.train(path);
    rf.predict(rf.create_test(path2));
    
    return 0;
}
