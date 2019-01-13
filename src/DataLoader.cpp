//
// Created by Iván Canales Martín on 2019-01-08.
//

#include "DataLoader.h"
#include "utils.h"
#include "task1.h"

void DataLoader::addPath(cv::String path, int label) {
    this->paths.push_back(path);
    this->labels.push_back(label);
}

std::tuple<cv::Mat, cv::Mat> DataLoader::load(cv::String train_path, int nClasses, cv::HOGDescriptor& hog) {
    // LOAD DATASET
    cv::Mat trainLabel;
    // taking images name
    for (int lab = 0; lab < nClasses ; lab++) {
        cv::String path(train_path + std::to_string(lab) + "/");
        this->labels.push_back(lab);
        this->paths.push_back(path);
    }

    std::vector<std::string> v;
    std::vector<std::string> v2;

    for (int i = 0; i < paths.size(); i++) {
        v2 = read_directory(this->paths[i]);
        v.insert(v.end(), v2.begin(), v2.end());
        for (size_t j = 0; j < v2.size(); j++) {
            trainLabel.push_back(labels[i]);
        }
    }


    cv::Mat trainData((int)v.size(),(int)hog.getDescriptorSize(),CV_32F);
    // converting in Mat
    cv::Mat image, editedImage, grayImg;
    std::vector<float> descriptor;
    int iter = 0;
    for (auto &i : v) {
        image = cv::imread(i, cv::IMREAD_COLOR);

        cv::resize(image, editedImage, cv::Size(WIN_SIZE,WIN_SIZE)); // Check later if it's correct!
        cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

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
