//
// Created by Iván Canales Martín on 2019-01-04.
//

//
//  RandomForest.cpp
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 03/01/2019.
//  Copyright © 2019 Iván Canales Martín. All rights reserved.
//
//#include "RandomForest.hpp"
#include <regex>
#include <numeric>
#include "utils.h"
#include "task1.h"
#include "RandomForest.hpp"
#include "DataLoader.h"

namespace ml = cv::ml;

RandomForest::RandomForest(int n, int samples, cv::HOGDescriptor &hog, int mc, int md, int ms, int f) {
    TreePtr tree;
    this->numTrees = n;
    for (int i = 0; i < this->numTrees; i++)
    {
        tree = cv::ml::DTrees::create();
        tree->setCVFolds(f);
        tree->setMaxCategories(mc);
        tree->setMaxDepth(md);
        tree->setMinSampleCount(ms);
        this->dtrees.push_back(tree);
    }

    this->nClasses = mc;
    this->hog = hog;
    this->nsample = samples;
}

void RandomForest::train(cv::Mat &train_features, cv::Mat &train_label) {
    int n = this->nsample;
    if (this->nsample == RandomForest::ALL_SAMPLES) n = train_features.rows;
    // Create samples
    cv::Mat sampleFeatures(n, train_features.cols, train_features.type());
    cv::Mat sampleLabels(n, 1, CV_32S);
    std::vector<int> vec(n);
    for (size_t i = 0; i < this->numTrees; i++)
    {
        std::cout << i+1 << '/' << this->numTrees << std::endl;
        // Sampling
        vec =  randomvec(0,train_features.rows, n);
        for (int j,k = 0; k < n; k++) {
            j = vec[k];
            train_features.row(j).copyTo(sampleFeatures.row(k));
            (*sampleLabels.ptr<int>(k)) = (*train_label.ptr<int>(j));
        }
        // train tree
        this->dtrees[i]->train(cv::ml::TrainData::create(sampleFeatures, cv::ml::ROW_SAMPLE, sampleLabels));
    }
}

std::vector<float> RandomForest::predict(cv::Mat descriptor) {
    // create_test(test_path);
    cv::Mat f = descriptor.reshape(1,1);

    std::vector<int> classes(this->nClasses);
    // predictions contains all the predictions, for each tree(row) for each sample(col)

    std::fill(classes.begin(),classes.end(),0);
    float p;
    cv::Mat res;
    for (auto tree: this->dtrees) {
        p = tree->predict(f);
        classes[p]++;
    }

    std::vector<float> out(this->nClasses);
    float sum = 0;
    for (auto& n : classes) {
        sum += n;
    }

    for (int j = 0; j < this->nClasses; j++) {
        out[j] = (float) classes[j] / sum;
    }

    return out;
}

std::vector<float> RandomForest::predictImage(cv::Mat images) {
    cv::Mat editedImage, grayImage, m;
    std::vector<float> descriptor;

    cv::resize(images, editedImage, cv::Size(WIN_SIZE, WIN_SIZE));
    cv::cvtColor(editedImage, grayImage, cv::COLOR_RGB2GRAY);

    this->hog.compute(editedImage, descriptor);

    cv::Mat s = cv::Mat(descriptor);
    return predict(s);
}

void RandomForest::save(std::string path) {
    std::ostringstream stream;
    int counter = 0;
    for (auto tree: dtrees) {
        stream.str(std::string());
        stream << path << "tree_" << counter++ << ".RandomForest";
        tree->save(stream.str());
    }
}

void RandomForest::load(std::string path) {
    std::vector<std::string> names = read_directory(path);
    std::regex r(".*tree_[[:digit:]]+\\.RandomForest$");
    this->dtrees.clear();
    for (std::string name: names) {

        if (std::regex_match(name, r)) {
            dtrees.push_back(ml::DTrees::load(name));
        }
    }

}