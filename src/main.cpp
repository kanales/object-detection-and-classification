//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <string>

#include "config.h"

#define XD lol

int main(int argc, const char * argv[]) {
    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    
    if( argc > 1)
    {
        imageName = argv[1];
    }
    
    cv::Mat image;
    image = cv::imread(imageName, cv::IMREAD_COLOR);
    
    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.

    cv::imshow("Display window", image);
    
    cv::imwrite($ROOT "output/test.jpg", image);
    cv::waitKey(0);
    return 0;
}
