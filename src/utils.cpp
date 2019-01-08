//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "utils.h"

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

template <class T>
void print_vector(std::vector<T> v) {
    std::cout << '[';
    for (T x: v) {
        std::cout << '\t' << x;
    }
    std::cout << " ]";
}


std::vector<std::string> read_directory(const std::string& name)
{
    std::vector<std::string> v;
    DIR* dirp = opendir(name.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) {
        v.push_back(name + dp->d_name);
    }
    closedir(dirp);
    v.erase(std::remove(v.begin(), v.end(), name + "."), v.end());
    v.erase(std::remove(v.begin(), v.end(), name + ".."), v.end());
    return v;
}

void print_vector(const std::vector<float> &vec) {
    for (float el: vec) {
        std::cout << el << ' ';
    }
}

// building a fixed HOG descriptor
cv::HOGDescriptor mk_hog() {
    cv::HOGDescriptor hog;
    hog.winSize = cv::Size(WIN_SIZE,WIN_SIZE);
    hog.blockSize = cv::Size(2*CELL_SIZE,2*CELL_SIZE);
    hog.blockStride = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.cellSize = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.nbins = NBINS;
    return hog;
}
