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
#include <filesystem>

#include "hog_visualization.h"
#include "config.h"
#include "utils.h"

std::vector<float> task1(cv::String imageName){
  cv::Mat image, editedImage, grayImg;
  std::vector<float> descriptor;
  image = cv::imread(imageName, cv::IMREAD_COLOR);

  if( image.empty() )                      // Check for invalid input
  {
      std::cout <<  "Could not open or find the image " << imageName << std::endl ;
  }
  else{
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.

    // cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
    resize(image, editedImage, cv::Size(image.cols * 5, image.rows * 5));
    // cv::imshow("Display window", editedImage);
    //
    cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

    // HOG descriptor
    cv::HOGDescriptor hog;
    // hog.winSize = grayImg.size();
    //    std::vector<cv::Point> positions;
    //    positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));

    hog.compute(grayImg,descriptor);
  }
  return descriptor;
}

int main(int argc, const char * argv[]) {

    cv::String imageName( $ROOT "/data/task1/obj1000.jpg" );
    cv::String path( $ROOT "/data/task2/train/00/" );
    std::vector<std::string> v;
    std::vector<std::string> v2;
    int nsample = 200;

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
    std::vector<float> descriptor = task1("/home/ale/magistrale/tracking/object-detection-and-classification/data/task2/train/00/0017.jpg");

    // /home/ale/magistrale/tracking/object-detection-and-classification/data/task2/train/00/0017.jpg

    cv::Mat trainData(nsample,descriptor.size(), CV_32F);
    cv::Mat trainLabel;
    cv::Mat testData(nsample,descriptor.size(), CV_32F);
    int iter = 0;
    std::vector<std::string> vfinal;
    std::vector<int> trainlab;
    for (int lab = 0; lab < 6; lab++) {
      cv::String path( $ROOT "/data/task2/train/0" + std::to_string(lab) + "/");
      std::vector<std::string> v2;
      read_directory(path,v2);
      v.insert( v.end(), v2.begin(), v2.end() );
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
    for (auto &i: vfinal){
      std::cout << i << '\n';
      cv::Mat m = cv::Mat(task1(i)).t();
      // m.convertTo( m, CV_32F );
      trainData.row(iter).copyTo(m);
      iter++;
    }

    cv::String path2( $ROOT "/data/task2/test/01/" );
    read_directory(path2,v2);
    iter = 0;
    for (auto &i: v2){
      std::cout << i << '\n';
      cv::Mat m = cv::Mat(task1(i)).t();
      // m.convertTo( m, CV_32F );
      testData.row(iter).copyTo(m);
      iter++;
    }

    cv::Ptr<cv::ml::DTrees> dtree = cv::ml::DTrees::create();

    dtree->setMaxDepth(16);
    dtree->setMinSampleCount(10);
    dtree->setMaxCategories(6);
    dtree->setCVFolds(0 /*10*/); // nonzero causes core dump
    // printmat(trainData);
    dtree->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));

    std::cout << dtree->predict(testData) << '\n';

    return 0;
}
