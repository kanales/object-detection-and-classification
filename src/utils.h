#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <iostream>
#include <sys/types.h>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

#define WIN_SIZE 96

typedef std::vector<std::string> stringvec;

// building a fixed HOG descriptor
cv::HOGDescriptor mk_hog(int cellSize = 12, int nbins = 9, int winSize = WIN_SIZE, int blockSize = 24,
        int cellStride = 12, bool signedGradient = false);
std::vector<int> randomvec(int range_min, int range_max, int n);
void printmat(cv::Mat M);
template <class T>
void print_vector(std::vector<T> v);
std::vector<std::string> read_directory(const std::string& name);
void print_vector(const std::vector<float> &vec);

#endif
