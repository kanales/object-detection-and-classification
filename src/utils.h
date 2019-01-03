#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <iostream>
#include <sys/types.h>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <iterator>

typedef std::vector<std::string> stringvec;

std::vector<int> randomvec(int range_min, int range_max, int n){
    std::random_device rd; // seed for PRNG
    std::mt19937 mt_eng(rd()); // mersenne-twister engine initialised with seed
    std::vector<int> v;
    // uniform distribution for generating random integers in given range
    std::uniform_int_distribution<> dist(range_min, range_max-1);

    for (int i = 0; i < n; ++i)
         v.push_back(dist(mt_eng));
    return v;
}


void printmat(cv::Mat M) {
  std::cout << "M = "<< std::endl << " "  << cv::format(M, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}

void printvec(stringvec v) {
  // std::copy(v.begin(), v.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  for (auto &i: v){
    std::cout << i << " " << std::endl;
  }
  std::cout << std::endl;
}

void read_directory(const std::string& name, stringvec& v)
{
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(name + dp->d_name);
    }
    closedir(dirp);
    v.erase(std::remove(v.begin(), v.end(), name + "."), v.end());
    v.erase(std::remove(v.begin(), v.end(), name + ".."), v.end());
}

#endif
