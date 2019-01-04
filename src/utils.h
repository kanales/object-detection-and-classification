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

typedef std::vector<std::string> stringvec;

std::vector<int> randomvec(int range_min, int range_max, int n);


void printmat(cv::Mat M);
template <class T>
void print_vector(std::vector<T> v);


void read_directory(const std::string& name, stringvec& v);

#endif
