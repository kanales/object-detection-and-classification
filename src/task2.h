#ifndef TASK2_H
#define TASK2_H

#include <opencv2/opencv.hpp>

#include "utils.h"
#include "RandomForest.hpp"
#include "config.h"
#include "task1.h"

typedef int Class;

// generate a vector of images given a directory path and is label (in a format "0"+label)
std::vector<cv::Mat> load_test(cv::String test_path, char val);

// execute task 2
float part2(int param);

#endif
