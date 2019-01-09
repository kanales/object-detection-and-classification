//
// Created by Iván Canales Martín on 2019-01-08.
//

#ifndef OBJECT_DETECTION_AND_CLASSIFICATION_DATALOADER_H
#define OBJECT_DETECTION_AND_CLASSIFICATION_DATALOADER_H


//#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

class DataLoader {
    std::vector<int> labels;
    std::vector<cv::String> paths;
public:
    void addPath(cv::String path, int label);
    std::tuple<cv::Mat, cv::Mat> load(cv::String path, int nClasses, cv::HOGDescriptor &hog);
};


#endif //OBJECT_DETECTION_AND_CLASSIFICATION_DATALOADER_H
