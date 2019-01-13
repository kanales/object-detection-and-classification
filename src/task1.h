#ifndef TASK1_H
#define TASK1_H

#include <opencv2/opencv.hpp>

#include "utils.h"
#include "config.h"
#include "hog_visualization.h"

// function that given an image extract his descriptor and visualize it
void show_descriptor(cv::HOGDescriptor &hog, cv::String imageName, bool showImage);

// function that given an image extract his descriptor
std::vector<float> extract_descriptor(cv::HOGDescriptor& hog, cv::String imageName);

// execute task 1
// maybe has to be more interactive with the user
void part1(int image);
#endif
