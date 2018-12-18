//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

#include <iostream>
#include <string>

#include "hog_visualization.h"
#include "config.h"

int main(int argc, const char * argv[]) {
    cv::String imageName( $ROOT "data/task1/obj1000.jpg" );
    
    if( argc > 1)
    {
        imageName = argv[1];
    }
    
    cv::Mat image, editedImage, grayImg;
    image = cv::imread(imageName, cv::IMREAD_COLOR);
    
    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.

    //cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
    
    cv::imshow("Display window", editedImage);
    
    cv::cvtColor(editedImage, grayImg, CV_RGB2GRAY);
    
    // HOG descriptor
    cv::HOGDescriptor hog;
    hog.winSize = grayImg.size();
//    std::vector<cv::Point> positions;
//    positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));
    std::vector<float> descriptor;
    
    hog.compute(grayImg,descriptor);
    
    visualizeHOG(grayImg, descriptor, hog);
    
    cv::imwrite($ROOT "output/test.jpg", image);
    cv::waitKey(0);
    return 0;
}

