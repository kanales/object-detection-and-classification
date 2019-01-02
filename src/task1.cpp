//
//  task1.cpp
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 18/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//

#include "task1.hpp"

int main(int argc, const char * argv[]) {
    cv::String imageName($ROOT "data/task1/obj1000.jpg");
    if( argc > 1)
    {
        imageName = argv[1];
        // string imageName = "./data/task1/obj1000.jpg";
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
    hog.blockSize = cv::Size(32,32);
    hog.blockStride = cv::Size(16,16);
    hog.cellSize = cv::Size(16,16);
    hog.nbins = 8;
    std::vector<cv::Point> positions;
    //    positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));
    std::vector<float> descriptor;
    
    hog.compute(grayImg,descriptor);
    
    visualizeHOG(editedImage, descriptor, hog);
    
    return 0;
}
