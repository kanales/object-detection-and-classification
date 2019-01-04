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
#include "utils.h"
#include "task1.h"
#include "RandomForest.hpp"

RandomForest::RandomForest(int n, int samples, cv::HOGDescriptor& hog, int mc, int f, int md, int ms) {
    TreePtr tree;
    numTrees = n;
    for (int i = 0; i < numTrees; i++)
    {
        tree = cv::ml::DTrees::create();
        tree->setCVFolds(f);
        tree->setMaxCategories(mc);
        tree->setMaxDepth(md);
        tree->setMinSampleCount(ms);
        dtrees.push_back(tree);
    }

    this->nClasses = mc;
    this->hog = hog;
    nsample = samples;

}

void RandomForest::setCVFolds(int val){
    for (size_t i = 0; i < numTrees; i++) {
        dtrees[i]->setCVFolds(val /*10*/); // nonzero causes core dump
    }
}

void RandomForest::setMaxCategories(int val){
    for (size_t i = 0; i < numTrees; i++) {
        dtrees[i]->setMaxCategories(val);
    }
}

void RandomForest::setMaxDepth(int val){
    for (size_t i = 0; i < numTrees; i++) {
        dtrees[i]->setMaxDepth(val );
    }
}

void RandomForest::setSampleCount(int val){
    for (size_t i = 0; i < numTrees; i++) {
        dtrees[i]->setMinSampleCount(val );
    }
}

// this shouldn't have to load the dataset
void RandomForest::train(cv::String train_path){
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
    int iter = 0;
    for (auto &i : v) {
        cv::Mat m = cv::Mat(extract_descriptor(hog, i)).t();
        // m.convertTo( m, CV_32F );
        m.copyTo(trainData.row(iter));
        //            trainData.row(iter).copyTo(m);

        iter++;
    }
    int n = nsample;
    if (nsample == -1) n = trainData.rows;
    // Create samples
    cv::Mat sampleFeatures(n, trainData.cols, trainData.type());
    cv::Mat sampleLabels(n, 1, CV_32S);
    std::vector<int> vec(n);
    for (size_t i = 0; i < numTrees; i++)
    {
        std::cout << i+1 << '/' << numTrees << std::endl;

        // Sampling
        vec =  randomvec(0,trainData.rows, n);
        for (int j,k = 0; k < n; k++) {
            j = vec[k];
            trainData.row(j).copyTo(sampleFeatures.row(k));
            (*sampleLabels.ptr<int>(k)) = (*trainLabel.ptr<int>(j));
        }
        // train tree
        dtrees[i]->train(cv::ml::TrainData::create(sampleFeatures, cv::ml::ROW_SAMPLE, sampleLabels));
    }
}

std::vector<float> RandomForest::predict(cv::Mat descriptor) {
    // create_test(test_path);
    cv::Mat f = descriptor.reshape(1,1);
    f.convertTo(f, CV_32F);

    std::vector<int> vout;
    std::vector<int> classes(nClasses);
    // predictions contains all the predictions, for each tree(row) for each sample(col)

    std::fill(classes.begin(),classes.end(),0);

    for (auto tree: dtrees) {
        classes[tree->predict(f)]++;
    }

    std::vector<float> out(nClasses);
    float sum = 0;
    for (auto& n : classes) {
        sum += n;
    }

    for (int j = 0; j < nClasses; j++) {
        out[j] = classes[j] / sum;
    }

    return out;
}

cv::Mat RandomForest::imageToSample(cv::Mat images) {
    cv::Mat editedImage, grayImage, m;
    std::vector<float> descriptor;

    cv::resize(images, editedImage, cv::Size(128,128));
    cv::cvtColor(editedImage, grayImage, cv::COLOR_RGB2GRAY);
    hog.compute(grayImage,descriptor);
    return cv::Mat(descriptor);
}

std::vector<float> RandomForest::predictImage(cv::Mat images) {
    return predict(imageToSample(images));
}
