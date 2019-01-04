#ifndef TASK1_H
#define TASK1_H

#define CELL_SIZE 16
#define NBINS 8
#define WIN_SIZE 128

#include <opencv2/opencv.hpp>

#include "config.h"
#include "hog_visualization.h"

const int N_FEATURES = (WIN_SIZE/CELL_SIZE);

// building a fixed HOG descriptor
cv::HOGDescriptor mk_hog() {
    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(WIN_SIZE,WIN_SIZE);
    hog.blockSize = cv::Size(2*CELL_SIZE,2*CELL_SIZE);
    hog.blockStride = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.cellSize = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.nbins = NBINS;
    return hog;
}

// function that given an image extract his descriptor and visualize it
void show_descriptor(cv::HOGDescriptor& hog, cv::String imageName) {
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    image = cv::imread(imageName, cv::IMREAD_COLOR);

    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image " << imageName << std::endl ;
    }
    else {
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

// function that given an image extract his descriptor
std::vector<float> extract_descriptor(cv::HOGDescriptor& hog, cv::String imageName) {
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    image = cv::imread(imageName, cv::IMREAD_COLOR);

    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image " << imageName << std::endl ;
    }
    else{
        cv::resize(image, editedImage, cv::Size(128,128)); // Check later if it's correct!
        cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

        hog.compute(grayImg,descriptor);
    }
    return descriptor;
}

// execute task 1
// maybe has to be more interactive with the user
void part1(int argc, const char *argv[]) {
  cv::String imageName( $ROOT "data/task1/obj1000.jpg" );

  if(argc > 1)
  {
    imageName = argv[1];
  }
  // TASK1
  cv::HOGDescriptor hog = mk_hog();

  std::cout << "Building descriptor and visualizing the descriptor (press any key to continue...)" << std::endl;
  show_descriptor(hog, imageName);
}

#endif
