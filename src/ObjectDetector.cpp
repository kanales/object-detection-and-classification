#include <utility>

//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "ObjectDetector.h"
#include "utils.h"

// controlClass determines which class will be considered as background and therefore ignored for detection purposes
ObjectDetector::ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride, float bgCutoff)
        : rf(rf) {
    this->rf = rf;
    this->nothingClass = backgroundClass;
    this->windowStride = windowStride;
    this->bgCutoff = bgCutoff;
    this->winSize = rf.hog.winSize;
}

std::vector<cv::Rect>
ObjectDetector::generateWindows(cv::Mat image, cv::Size winSize, float scaleFactor, int iterations) {
    std::vector<cv::Rect> out;

    // sliding window size
    //int windows_rows = 20;
    //int windows_cols = 20;
    winSize = rf.hog.winSize;
    for (int col = 0; col <= image.cols - winSize.width; col += windowStride){
        for (int row = 0; row <= image.rows - winSize.height; row += windowStride){
            cv::Rect window({col, row}, winSize);
            out.push_back(window);
            //  cv::Mat window_image = cv::Mat(image,window); // window content in Mat format
        }
    }
    winSize = {(int) std::ceil(winSize.width * scaleFactor), (int) std::ceil(winSize.height*scaleFactor)};


    return out;
}



std::tuple<Class, float> ObjectDetector::detectClass(cv::Rect rect, cv::Mat img) {

    //TODO test this
    // cv::Mat subimg = cv::Mat(img,rect);
    cv::Mat subimg = img(rect);
    std::vector<float> preds = this->rf.predictImage(subimg);
    int c = (int) std::distance(preds.begin(), std::max_element(preds.begin(),preds.end()));
    float probBG = preds[nothingClass];
    float confidence = preds[c];
    if(confidence < 0.5)
      return {nothingClass, probBG};

    std::cout << c << std::endl;
    return {c, confidence};
}

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat image) {

    std::vector<cv::Rect> windows = this->generateWindows(image, image.size(), 0.8, 1);

    int countBack = 0;
    int countObj = 0;

    // filter out background
    std::vector<DetectedObject> objs;


    for (const auto &win: windows) {
        auto [cls, confidence] = detectClass(win, image);

        if(cls==nothingClass)
          countBack++;
        else
          countObj++;


        if (cls != nothingClass) {

            DetectedObject obj {
                    cls,
                    win,
                    confidence
            };

            objs.push_back(obj);
        }
    }

    return objs;
}

std::vector<DetectedObject> ObjectDetector::nonMaximaSupression(std::vector<DetectedObject> objs) {
    return objs;
}
