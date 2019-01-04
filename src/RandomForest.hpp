#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
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
    cv::HOGDescriptor hog;

    // used for predicting from a list of images
    cv::Mat imageToSample(cv::Mat images);
public:
    RandomForest(int n, int samples, cv::HOGDescriptor& hog, int mc, int f=0, int md=20, int ms=10);

    void setCVFolds(int val);

    void setMaxCategories(int val);

    void setMaxDepth(int val);

    void setSampleCount(int val);

    // putting the right training data and the train path can be chosen before (we use it multiple times) (we have to do it differently)
    void train(cv::String train_path);
    std::vector<float> predict(cv::Mat sample);
    std::vector<float> predictImage(cv::Mat images);

};

#endif
