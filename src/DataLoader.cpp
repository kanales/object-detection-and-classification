//
// Created by Iván Canales Martín on 2019-01-08.
//

#include "DataLoader.h"
#include "utils.h"
#include "task1.h"

std::tuple<cv::Mat, cv::Mat> DataLoader::load(cv::String train_path, int nClasses, cv::HOGDescriptor &hog) {
    // LOAD DATASET
    std::vector<std::string> v;
    cv::Mat trainLabel;
    // taking images name
    for (int lab = 0; lab < nClasses ; lab++) {
        cv::String path(train_path + std::to_string(lab) + "/");
        std::vector<std::string> v2;
        read_directory(path, v2);
        v.insert(v.end(), v2.begin(), v2.end());
        // index vector
        for (size_t j = 0; j < v2.size(); j++) {
            trainLabel.push_back(lab);
        }
    }
    cv::Mat trainData((int)v.size(),(int)hog.getDescriptorSize(),CV_32F);
    // converting in Mat
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    int iter = 0;
    for (auto &i : v) {
        image = cv::imread(i, cv::IMREAD_COLOR);

        cv::resize(image, editedImage, cv::Size(128,128)); // Check later if it's correct!
        //cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

        hog.compute(editedImage,descriptor);

        cv::Mat m;
        m = cv::Mat(extract_descriptor(hog, i)).t();
        // m.convertTo( m, CV_32F );
        m.copyTo(trainData.row(iter));
        //            trainData.row(iter).copyTo(m);

        iter++;
    }

    return {trainData, trainLabel};
}
