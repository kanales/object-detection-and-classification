#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#ifdef VERBOSE
#include <stdio>
#endif

#include "task1.h"

#include <algorithm>
typedef cv::Ptr<cv::ml::DTrees> TreePtr;

class random_forest{
private:
    int numTrees;
    std::vector<TreePtr> dtrees;
    int folds;
    int maxCategories;
    int maxDepth;
    int minSampleCount;
    int nsample;
    int nClasses = 6;
    cv::HOGDescriptor hog;
public:
    random_forest(int n, int samples, cv::HOGDescriptor hog, int mc, int f=0, int md=10, int ms=16) {
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
        this->hog = hog;
        nsample = samples;
    }
    
    void setCVFolds(int val){
        for (size_t i = 0; i < numTrees; i++) {
            dtrees[i]->setCVFolds(val /*10*/); // nonzero causes core dump
        }
    }
    
    void setMaxCategories(int val){
        for (size_t i = 0; i < numTrees; i++) {
            dtrees[i]->setMaxCategories(val);
        }
    }
    
    void setMaxDepth(int val){
        for (size_t i = 0; i < numTrees; i++) {
            dtrees[i]->setMaxDepth(val );
        }
    }
    
    void setSampleCount(int val){
        for (size_t i = 0; i < numTrees; i++) {
            dtrees[i]->setMinSampleCount(val );
        }
    }
    
    // putting the right training data and the train path can be chosen before (we use it multiple times) (we have to do it differently)
    void train(cv::String train_path){
        // LOAD DATASET
        std::vector<std::string> v;
        cv::Mat trainLabel;
        // taking images name
        for (int lab = 0; lab < 6; lab++) {
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
        
        // Create samples
        cv::Mat sampleFeatures(nsample, trainData.cols, trainData.type());
        cv::Mat sampleLabels(nsample, 1, CV_32S);
        std::vector<int> vec(nsample);
        for (size_t i = 0; i < numTrees; i++)
        {
            std::cout << i+1 << '/' << numTrees << std::endl;
            
            // Sampling
            vec =  randomvec(0,trainData.rows, nsample);
            for (int j,k = 0; k < nsample; k++) {
                j = vec[k];
                trainData.row(j).copyTo(sampleFeatures.row(k));
                (*sampleLabels.ptr<int>(k)) = (*trainLabel.ptr<int>(j));
            }
            // train tree
            dtrees[i]->train(cv::ml::TrainData::create(sampleFeatures, cv::ml::ROW_SAMPLE, sampleLabels));
        }
    }
    
    std::vector<float> predict(cv::Mat sample) {
        // create_test(test_path);
        cv::Mat f = sample.reshape(1,1);
        f.convertTo(f, CV_32F);
        
        std::vector<int> vout;
        std::vector<int> classes(nClasses);
        // predictions contains all the predictions, for each tree(row) for each sample(col)
        
        std::fill(classes.begin(),classes.end(),0);
        
        for (int j = 0; j < numTrees; j++) {
            int k = dtrees[j]->predict(f);
            classes[k]++;
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
//        return (int)std::distance(
//                classes.begin(),
//                std::max_element(classes.begin(),classes.end()));
    }
};

#endif
