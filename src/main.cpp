//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/opencv.hpp>
// #include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/objdetect.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "hog_visualization.h"
#include "config.h"
#include "utils.h"
#include "task1.h"
#include "random_forest.h"

int main(int argc, const char * argv[]) {

    cv::String imageName( $ROOT "/data/task1/obj1000.jpg" );
    cv::String path( $ROOT "/data/task2/train/00/" );
    std::vector<std::string> v;
    std::vector<std::string> v2;
    int nsample = 100;

    // read_directory(path,v);
    // printvec(v);

    if( argc > 1)
    {
        imageName = argv[1];
        // string imageName = "./data/task1/obj1000.jpg";
    }



    // descriptor = task1(imageName);

    // visualizeHOG(grayImg, descriptor, hog);
    //
    // cv::imwrite( $ROOT "output/test.jpg", grayImg);
    // cv::waitKey(0);

    // TASK2
    std::vector<float> descriptor = task1("/home/ale/magistrale/tracking/object-detection-and-classification/data/task2/train/00/0011.jpg");

    // /home/ale/magistrale/tracking/object-detection-and-classification/data/task2/train/00/0017.jpg
    std::cout << descriptor.size() << '\n';
    return 0;
    cv::Mat trainData(nsample,descriptor.size(), CV_32F);
    cv::Mat trainLabel;
    cv::Mat testData(nsample,descriptor.size(), CV_32F);
    int iter = 0;
    std::vector<std::string> vfinal;
    std::vector<int> trainlab;
	// taking images name
    for (int lab = 0; lab < 6; lab++) {
      cv::String path( $ROOT "/data/task2/train/0" + std::to_string(lab) + "/");
      std::vector<std::string> v2;
      read_directory(path,v2);
      v.insert( v.end(), v2.begin(), v2.end() );
	  // index vector
      for (size_t j = 0; j < v2.size(); j++) {
        trainlab.push_back(lab);
      }
    }
    std::vector<int> randid = randomvec(0,v.size(),nsample);
    for (size_t i = 0; i < nsample; i++) {
      int x = randid[i];
      vfinal.push_back(v[x]);
      trainLabel.push_back(trainlab[x]);
    }
	// converting in Mat
    for (auto &i: vfinal){
      std::cout << i << '\n';
      cv::Mat m = cv::Mat(task1(i)).t();
      // m.convertTo( m, CV_32F );
      trainData.row(iter).copyTo(m);
      iter++;
    }
	// test data
    cv::String path2( $ROOT "/data/task2/test/02/" );
    read_directory(path2,v2);
    iter = 0;
    for (auto &i: v2){
      std::cout << i << '\n';
      cv::Mat m = cv::Mat(task1(i)).t();
      // m.convertTo( m, CV_32F );
      testData.row(iter).copyTo(m);
      iter++;
    }
    //
    // cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();
    //
    // dtree->setMaxDepth(16);
    // dtree->setMinSampleCount(10);
    // dtree->setMaxCategories(6);
    // dtree->setCVFolds(0 /*10*/); // nonzero causes core dump
    // // printmat(trainData);
    // dtree->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));
    //
    // std::cout << dtree->predict(testData) << '\n';

    // random_forest rf;


    return 0;
}
