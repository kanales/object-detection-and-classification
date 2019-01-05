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

// Update so it uses changes in scale?
std::vector<cv::Rect>
ObjectDetector::generateWindows(cv::Mat image, cv::Size winSize) {
    std::vector<cv::Rect> out;

    // sliding window size
    //int windows_rows = 20;
    //int windows_cols = 20;
    for (int col = 0; col <= image.cols - winSize.width; col += windowStride){
        for (int row = 0; row <= image.rows - winSize.height; row += windowStride){
            cv::Rect window({col, row}, winSize);
            out.push_back(window);
            //  cv::Mat window_image = cv::Mat(image,window); // window content in Mat format
        }
    }
    //winSize = {(int) std::ceil(winSize.width * scaleFactor), (int) std::ceil(winSize.height*scaleFactor)};


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
    // in case of doubt...
    if(confidence < 0.7)
      return {nothingClass, bgCutoff};

    return {c, confidence};
}

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat image) {

    std::vector<cv::Rect> windows = this->generateWindows(image, rf.hog.winSize); // 1:1 window size

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

    return nonMaximaSupression(objs);
}

// todo finish
std::vector<DetectedObject> ObjectDetector::nonMaximaSupression(std::vector<DetectedObject> &objs) {
    std::vector<DetectedObject> out;
    //sort by confidence
    std::sort(objs.begin(), objs.end(), [](DetectedObject a, DetectedObject b) {
        // sort into descending confidences
        return a.confidence > b.confidence;
    });

    auto end   = objs.end();
    auto begin = objs.begin();
    while (begin != end) {
        DetectedObject o = *begin; // get element with most confidence
        std::cout << o.cls;
        out.push_back(o);
        end = std::remove_if(begin, end, [&o] (DetectedObject el) {
            //todo adjust threshold
            float thr = 0.25; // For now delete if intersects and
            float ratio = (float)(el.rect & o.rect).area() / (el.rect | o.rect).area();
            return (el.cls == o.cls) && (ratio > thr);
        });
    }
    return out;
}
