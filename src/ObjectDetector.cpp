#include <utility>

//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "ObjectDetector.h"
#include "utils.h"

// controlClass determines which class will be considered as background and therefore ignored for detection purposes
ObjectDetector::ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride)
        : rf(rf) {
    this->rf = rf;
    this->nothingClass = backgroundClass;
    this->windowStride = windowStride;
}

std::vector<cv::Rect> ObjectDetector::generateWindows(cv::Mat image) {
    std::vector<cv::Rect> out;

    // sliding window size
    int windows_rows = 20;
    int windows_cols = 20;
    int StepSlide = 10; //change

    for (int col = 0; col <= image.cols - windows_cols; col += StepSlide){

      for (int row = 0; row <= image.rows - windows_rows; row += StepSlide){

        cv::Rect window(col, row, windows_rows, windows_cols);
        out.push_back(window);
      //  cv::Mat window_image = cv::Mat(image,window); // window content in Mat format
      }
    }

    return out;
}



Class ObjectDetector::detectClass(cv::Rect rect, cv::Mat img) {

    //TODO test this
    // cv::Mat subimg = cv::Mat(img,rect);
    cv::Mat subimg = img(rect);
    std::vector<float> preds = this->rf.predictImage(subimg);
    int c = (int) std::distance(preds.begin(), std::max_element(preds.begin(),preds.end()));
    float confidence = preds[c];
    if(confidence < 0.5)
      return -1;
    return c;

}

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat image) {

    std::vector<cv::Rect> windows = this->generateWindows(image);

    int countBack = 0;
    int countObj = 0;

    // filter out background
    std::vector<DetectedObject> objs;
    for (const auto &win: windows) {
        Class cls = detectClass(win, image);

        if(cls==1)
          countBack++;
        else
          countObj++;


        if (cls != nothingClass && cls != -1) {

            DetectedObject obj {
                    cls,
                    win
            };

            objs.push_back(obj);
        }
    }

    std::cout << "background " << countBack << '\n';
    std::cout << "object " << countObj << '\n';
    return objs;
}

std::vector<DetectedObject> ObjectDetector::nonMaximaSupression(std::vector<DetectedObject> objs) {
    return objs;
}
