#ifndef TASK1_H
#define TASK1_H

#define CELL_SIZE 16
#define NBINS 8
#define WIN_SIZE 128

#include <opencv2/opencv.hpp>

#include "config.h"
#include "hog_visualization.h"

cv::HOGDescriptor mk_hog();
void show_descriptor(cv::HOGDescriptor hog, cv::String imageName);
std::vector<float> extract_descriptor(cv::HOGDescriptor hog, cv::String imageName);

#endif
