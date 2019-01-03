//
//  ObjectDetector.h
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 03/01/2019.
//  Copyright © 2019 Iván Canales Martín. All rights reserved.
//

#ifndef ObjectDetector_h
#define ObjectDetector_h

#include <opencv2/opencv.hpp>
#include "RandomForest.hpp"
#include <vector>

typedef int Class;

struct DetectedObject {
    Class cls; // class of the object
    cv::Rect rect; // delimiting rect of the object
};

class ObjectDetector {
    int windowStride;
    Class nothingClass;
    RandomForest &rf;

    Class predictPatch(cv::Mat patch);
public:
    ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride = 1);
    std::vector<cv::Rect> generateWindows(cv::Mat image);
    Class detectClass(cv::Rect rect, cv::Mat img);
    std::vector<DetectedObject> nonMaximaSupression(std::vector<DetectedObject> objs);
    // Returns a list of
    std::vector<DetectedObject> detectObjects(cv::Mat image);
};

//

// controlClass determines which class will be considered as background and therefore ignored for detection purposes
ObjectDetector::ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride)
        : rf(rf) {
    this->rf = rf;
    this->nothingClass = backgroundClass;
    this->windowStride = windowStride;
}

std::vector<cv::Rect> ObjectDetector::generateWindows(cv::Mat image) {
    std::vector<cv::Rect> out;
    //TODO
    return out;
}

Class ObjectDetector::detectClass(cv::Rect rect, cv::Mat img) {
    //TODO
    cv::Mat subimg = img(rect);
    return 0;
}

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat image) {
    std::vector<cv::Rect> windows = this->generateWindows(image);

    // filter out background
    std::vector<DetectedObject> objs;
    for (auto win: windows) {
        Class cls = detectClass(win, cv::Mat());
        if (cls != nothingClass) {
            DetectedObject obj {
                cls,
                win
            };
            objs.push_back(obj);
        }
    }

    return objs;
}

std::vector<DetectedObject> ObjectDetector::nonMaximaSupression(std::vector<DetectedObject> objs) {
    return objs;
}

Class ObjectDetector::predictPatch(cv::Mat patch) {
    return 0;
}

#endif /* ObjectDetector_h */
