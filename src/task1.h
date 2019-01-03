#include <opencv2/opencv.hpp>

#include "config.h"

#ifndef TASK1_H
#define TASK1_H

#define CELL_SIZE 16
#define NBINS 8
#define WIN_SIZE 128

const int N_FEATURES = (WIN_SIZE/CELL_SIZE);

cv::HOGDescriptor mk_hog() {
    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(WIN_SIZE,WIN_SIZE);
    hog.blockSize = cv::Size(2*CELL_SIZE,2*CELL_SIZE);
    hog.blockStride = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.cellSize = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.nbins = NBINS;
    return hog;
}

void show_descriptor(cv::HOGDescriptor hog, cv::String imageName) {
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    image = cv::imread(imageName, cv::IMREAD_COLOR);
    
    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image " << imageName << std::endl ;
    }
    else {
        cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
        cv::resize(image, editedImage, cv::Size(WIN_SIZE,WIN_SIZE)); // Check later if it's correct!
        
        cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);
        std::cout << descriptor.size() << std::endl;
        // HOG descriptor
        
        hog.compute(grayImg,descriptor);
        // for task 1 execute this
        visualizeHOG(grayImg, descriptor, hog);
        cv::waitKey(0);
    }
}


std::vector<float> extract_descriptor(cv::HOGDescriptor hog, cv::String imageName) {
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    image = cv::imread(imageName, cv::IMREAD_COLOR);
    
    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image " << imageName << std::endl ;
    }
    else{
        
        //cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
        cv::resize(image, editedImage, cv::Size(128,128)); // Check later if it's correct!
        cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);
        
        hog.compute(grayImg,descriptor);
        
    }
    return descriptor;
}

#endif
