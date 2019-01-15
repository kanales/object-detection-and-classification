#include <utility>

//
// Created by Iván Canales Martín on 2019-01-03.
//

#include "ObjectDetector.h"
#include "utils.h"
#include "config.h"

// controlClass determines which class will be considered as background and therefore ignored for detection purposes
ObjectDetector::ObjectDetector(RandomForest &rf, Class backgroundClass, int windowStride, float bgCutoff, float overlap_thr)
        : rf(rf) {
    this->rf = rf;
    this->nothingClass = backgroundClass;
    this->windowStride = windowStride;
    this->bgCutoff = bgCutoff;
    this->winSize = rf.hog.winSize;
    this->overlap_thr = overlap_thr;
}

// Update so it uses changes in scale?
std::vector<cv::Rect>
ObjectDetector::generateWindows(cv::Mat image, cv::Size minWinSize, cv::Size maxWinSize, int steps) {
    std::vector<cv::Rect> out;

    // sliding window size
    //int windows_rows = 20;
    //int windows_cols = 20;
    float hfactor = (maxWinSize.height - minWinSize.height) / (float)steps;
    float wfactor = (maxWinSize.height - minWinSize.height) / (float)steps;
    cv::Size winSize = minWinSize;
    for (int i = 0; i < steps; i++) {
        winSize.height += hfactor;
        winSize.width  += wfactor;
        for (int col = 0; col <= image.cols - winSize.width; col += windowStride){
            for (int row = 0; row <= image.rows - winSize.height; row += windowStride){
                cv::Rect window({col, row}, winSize);
                out.push_back(window);
                //  cv::Mat window_image = cv::Mat(image,window); // window content in Mat format
            }
        }
    }

    //winSize = {(int) std::ceil(winSize.width * scaleFactor), (int) std::ceil(winSize.height*scaleFactor)};


    return out;
}



std::tuple<Class, float> ObjectDetector::detectClass(cv::Mat &img, float cutoff) {
    //TODO test this
    // cv::Mat subimg = cv::Mat(img,rect);
    //cv::Mat imagebw, subimg;
    //cv::cvtColor(img, imagebw, cv::COLOR_RGB2GRAY);
    //cv::resize(imagebw(rect), subimg, this->winSize);
    std::vector<float> preds = this->rf.predictImage(img);
    //int c = (int) std::distance(preds.begin(), std::max_element(preds.begin(),preds.end()));
    int maxi = this->nothingClass;
    float max = 0;
    for (int i = 0; i < preds.size(); i++) {
        if (preds[i] > max) {
            maxi = i;
            max  = preds[i];
        }
    }
    float confidence = preds[maxi];
    // in case of doubt...
    if(confidence < cutoff)
        return {nothingClass, cutoff};
    return {maxi, confidence};
}

/*std::vector<DetectedObject> ObjectDetector::detectAllObjects(cv::Mat image) {
    if constexpr(DEBUG) std::cout << "Generating windows...";
    std::vector<cv::Rect> windows = this->generateWindows(image, cv::Size(50,50), cv::Size(150,150), 10); // 1:1 window size
    if constexpr(DEBUG) std::cout << "DONE" << std::endl;
    int countBack = 0;
    int countObj = 0;

    // filter out background
    std::vector<DetectedObject> objs;

    if constexpr(DEBUG) std::cout << "Detecting objects...";
    for (const auto &win: windows) {
        auto [cls, confidence] = detectClass(win, image, 0);
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
    if constexpr(DEBUG) std::cout << "DONE" << std::endl;

    return nonMaximaSupression(objs);
}*/

std::vector<DetectedObject> ObjectDetector::detectObjects(cv::Mat &image, float resizeFactor, int steps) {
    if constexpr(DEBUG) std::cout << "Generating windows...";
    //std::vector<cv::Rect> windows = this->generateWindows(image, cv::Size(50,50), cv::Size(150,150), 10); // 1:1 window size
    cv::Mat grayImg = image;
    //cv::cvtColor(image, grayImg, CV_BGR2GRAY);
    cv::Size imgSize = {image.cols, image.rows};

    float currFactor = std::max(winSize.width / (float)image.cols, winSize.height / (float)image.rows);
    cv::Mat resizedImg;
    std::vector<DetectedObject> objs;
    for (int i = 0; i < steps; i++) {
        std::cout << i+1 <<'/'<< steps << std::endl;
        imgSize.width  = (int)std::ceil(image.cols * currFactor);
        imgSize.height = (int)std::ceil(image.rows * currFactor);

        cv::Size realWindowSize = {
                (int) (winSize.width    / currFactor),
                (int) (winSize.height   / currFactor),
        };

        //if (imgSize.height < winSize.height || imgSize.width < winSize.width ) break;

        cv::resize(grayImg, resizedImg, imgSize);

        for (int col = 0; col <= resizedImg.cols - winSize.width; col += windowStride){
            for (int row = 0; row <= resizedImg.rows - winSize.height; row += windowStride){
                cv::Rect resizedWindow({col, row}, winSize);

                cv::Rect realWindow({ (int)(col/currFactor), (int) (row /currFactor)}, realWindowSize);

                cv::Mat subimg = resizedImg(resizedWindow);
                auto [cls, confidence] = detectClass(subimg, this->bgCutoff);
                if (cls != nothingClass) {
                    DetectedObject obj {
                            cls,
                            realWindow,
                            confidence
                    };

                    objs.push_back(obj);
                }
                //windows.push_back({resizedWindow,realWindow});
                //  cv::Mat window_image = cv::Mat(image,window); // window content in Mat format
            }
        }

        currFactor *= resizeFactor;
    }


    if constexpr(DEBUG) std::cout << "DONE" << std::endl;

    // filter out background
    //std::vector<DetectedObject> objs;
    return nonMaximaSupression(objs);
}

// todo finish
std::vector<DetectedObject> ObjectDetector::nonMaximaSupression(std::vector<DetectedObject> &objs) {
    //sort by confidence
    std::sort(objs.begin(), objs.end(), [](DetectedObject a, DetectedObject b) {
        // sort into descending confidences
        return a.confidence > b.confidence;
    });

    auto end   = objs.end();
    auto begin = objs.begin();
    while (begin != end) {
        DetectedObject o = *begin; // get element with most confidence
        begin += 1; // displace beginning
        end = std::remove_if(begin, end, [this,&o] (DetectedObject el) {
            //todo adjust threshold
            float thr = this->overlap_thr; // For now delete if intersects and
            int area = (el.rect & o.rect).area();
            float ratio = area / (float)(el.rect | o.rect).area();
            return (ratio > thr);
            //return (area / (float) el.rect.area()) > thr || (area / (float) o.rect.area()) > thr;
        });
    }
    return std::vector<DetectedObject>(objs.begin(), end);
}
