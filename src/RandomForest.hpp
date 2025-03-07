#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

typedef cv::Ptr<cv::ml::DTrees> TreePtr;
class RandomForest{
private:
    int numTrees;
    std::vector<TreePtr> dtrees;
    int folds;
    int maxCategories; // not used?
    int maxDepth;
    int minSampleCount;
    int nsample;
    int nClasses;

public:
    cv::HOGDescriptor hog;
    static const int ALL_SAMPLES = -1;
    RandomForest(int n, int samples, cv::HOGDescriptor &hog, int mc, int md = 20, int ms = 100, int f = 0);
    // putting the right training data and the train path can be chosen before (we use it multiple times) (we have to do it differently)
    void train(cv::Mat &train_features, cv::Mat &train_label);
    std::vector<float> predict(cv::Mat sample);
    std::vector<float> predictImage(cv::Mat images);

    void save(std::string path);
    void load(std::string path);
};

#endif
