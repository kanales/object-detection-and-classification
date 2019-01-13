//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "task1.h"

// function that given an image extract his descriptor and visualize it
void show_descriptor(cv::HOGDescriptor &hog, cv::String imageName, bool showImage) {
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    image = cv::imread(imageName, cv::IMREAD_COLOR);

    if( image.empty() )                      // Check for invalid input
    {
        std::cout <<  "Could not open or find the image " << imageName << std::endl ;
    }
    else {
        cv::resize(image, editedImage, cv::Size(WIN_SIZE,WIN_SIZE)); // Check later if it's correct!

        //cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);
        //std::cout << descriptor.size() << std::endl;
        // HOG descriptor

        hog.compute(editedImage,descriptor);
        // for task 1 execute this
        //
        if (!showImage) editedImage = cv::Mat(WIN_SIZE, WIN_SIZE, CV_8UC3, cv::Scalar(0,0,0));
        visualizeHOG(editedImage, descriptor, hog);
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

        cv::resize(image, editedImage, cv::Size(WIN_SIZE,WIN_SIZE)); // Check later if it's correct!
        cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

        hog.compute(grayImg,descriptor);
    }
    return descriptor;
}

cv::String imageNames[] = {
        cv::String( $ROOT "data/task1/obj1000.jpg" ),
        cv::String( $ROOT "data/task1/obj2000.jpg" ),
        cv::String( $ROOT "data/task1/obj3000.jpg" ),
};

// execute task 1
// maybe has to be more interactive with the user
void part1(int image) {
    cv::String imageName = imageNames[image];

    // TASK1
    cv::HOGDescriptor hog = mk_hog();

    std::cout << "Building descriptor and visualizing the descriptor (press any key to continue...)" << std::endl;
    show_descriptor(hog, imageName, true);
}