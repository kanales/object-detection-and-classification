//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <string>

#include "hog_visualization.h"
#include "config.h"

#define CELL_SIZE 16
#define NBINS 8

int main(int argc, const char * argv[]) {
    int objHeight, objWidth;
    cv::Mat img1, paddedImg1, bwImg1, img2, paddedImg2, bwImg2;

    img1 = cv::imread($ROOT "data/task2/train/00/0000.jpg");
    img2 = cv::imread($ROOT "data/task2/train/01/0000.jpg");

    // padding
    objHeight = (int) (std::ceil((double)  std::max(img1.rows, img2.rows) / CELL_SIZE) * CELL_SIZE);
    objWidth  = (int) (std::ceil((double) std::max(img1.cols, img2.cols) / CELL_SIZE) * CELL_SIZE);
    
    cv::copyMakeBorder(img1, paddedImg1, objHeight - img1.rows , 0, objWidth - img1.cols, 0, IPL_BORDER_CONSTANT);
    cv::copyMakeBorder(img2, paddedImg2, objHeight - img2.rows , 0, objWidth - img2.cols, 0, IPL_BORDER_CONSTANT);

    cv::cvtColor(paddedImg1, bwImg1, CV_RGB2GRAY);
    cv::cvtColor(paddedImg2, bwImg2, CV_RGB2GRAY);

    // HOG descriptor
    cv::HOGDescriptor hog;
    hog.blockSize = cv::Size(2*CELL_SIZE,2*CELL_SIZE);
    hog.blockStride = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.cellSize = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.nbins = NBINS;

    std::vector<float> descriptor1, descriptor2;

    hog.winSize = bwImg1.size();
    hog.compute(bwImg1, descriptor1);

    hog.winSize = bwImg2.size();
    hog.compute(bwImg2, descriptor2);

    // Tree
    cv::Mat m, feats, labels; // start empty
    feats.push_back(cv::Mat(descriptor1).reshape(1,1));
    labels.push_back(1);
    feats.push_back(cv::Mat(descriptor2).reshape(1,1));
    labels.push_back(2);

    
    cv::Ptr<cv::ml::DTrees> model = cv::ml::DTrees::create();
    cv::Ptr<cv::ml::TrainData> bar = cv::ml::TrainData::create(feats, cv::ml::ROW_SAMPLE, labels);
    model->train(bar);
    
    return 0;
}
