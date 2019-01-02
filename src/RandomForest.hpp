//
//  RandomForest.hpp
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 18/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//

#ifndef RandomForest_hpp
#define RandomForest_hpp

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

class RandomForest {
    std::vector<cv::ml::DTrees> trees;
public:
    void create(const int ntrees);
    bool train(const cv::Ptr<cv::ml::TrainData>& trainData, int flags = 0);
    float predict(cv::InputArray samples, cv::OutputArray results=cv::noArray(), int flags=0);
};

#endif /* RandomForest_hpp */
