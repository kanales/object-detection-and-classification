#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>

typedef std::vector<std::string> stringvec;

std::vector<int> randomvec(int range_min, int range_max, int n);

void printmat(cv::Mat M);

void printvec(stringvec v);

void read_directory(const std::string& name, stringvec& v);

#endif
