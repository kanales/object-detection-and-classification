#include <utility>

//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "ObjectDetector.h"

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
    //TODO test this
    cv::Mat subimg = img(rect);
    std::vector<float> preds = rf.predictImage(subimg);
    return (int) std::distance(preds.begin(), std::max_element(preds.begin(),preds.end()));
}

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat image) {
    std::vector<cv::Rect> windows = this->generateWindows(std::move(image));

    // filter out background
    std::vector<DetectedObject> objs;
    for (const auto &win: windows) {
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