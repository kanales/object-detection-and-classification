#ifndef TASK3_H
#define TASK3_H

#include <opencv2/opencv.hpp>

#include <string>

#include "config.h"
#include "utils.h"

void image_rotation(cv::String imagePath, int angle);

void image_flip(cv::String imagePath);

void data_augmentation(cv::String train_path);

// execute task 3
void part3(bool retrain, float object_thr, float overlapthr);

void test(bool retrain, int image_index);

#endif
