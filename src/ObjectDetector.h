//
// Created by Iván Canales Martín on 2019-01-03.
//

#ifndef OBJECT_DETECTION_AND_CLASSIFICATION_OBJECTDETECTOR_H
#define OBJECT_DETECTION_AND_CLASSIFICATION_OBJECTDETECTOR_H

//
//  ObjectDetector.h
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 03/01/2019.
//  Copyright © 2019 Iván Canales Martín. All rights reserved.
//

#include <opencv2/opencv.hpp>
#include "RandomForest.hpp"
#include <vector>

typedef int Class;

struct DetectedObject {
    Class cls; // class of the object
    cv::Rect rect; // delimiting rect of the object
    float confidence;
};

class ObjectDetector {
    int windowStride;
    Class nothingClass;
    RandomForest &rf;
    cv::Size winSize;
public:
    float bgCutoff;
    float overlap_thr;
    ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride = 5, float bgCutoff = 0.60,
                       float overlap_thr = 0);
    std::vector<cv::Rect>
    generateWindows(cv::Mat image, cv::Size minWinSize, cv::Size maxWinSize, int steps);
    std::tuple<Class, float> detectClass(cv::Rect rect, cv::Mat img);
    std::vector<DetectedObject> nonMaximaSupression(std::vector<DetectedObject> &objs);
    // Returns a list of
    std::vector<DetectedObject> detectObjects(cv::Mat image);
};

#endif //OBJECT_DETECTION_AND_CLASSIFICATION_OBJECTDETECTOR_H
