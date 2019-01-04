#ifndef TASK1_H
#define TASK1_H

#define CELL_SIZE 16
#define NBINS 8
#define WIN_SIZE 128

#include <opencv2/opencv.hpp>

#include "config.h"
#include "hog_visualization.h"

// building a fixed HOG descriptor
cv::HOGDescriptor mk_hog();
// function that given an image extract his descriptor and visualize it
void show_descriptor(cv::HOGDescriptor& hog, cv::String imageName);

// function that given an image extract his descriptor
std::vector<float> extract_descriptor(cv::HOGDescriptor& hog, cv::String imageName);

// execute task 1
// maybe has to be more interactive with the user
void part1(int argc, const char *argv[]);
#endif
