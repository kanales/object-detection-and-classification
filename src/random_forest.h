#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include "task1.h"

#include <algorithm>

class random_forest{
private:
    int numTrees;
    std::vector<cv::Ptr<cv::ml::DTrees>> dtrees;
    int folds;
    int maxCategories;
    int maxDepth;
    int minSampleCount;
    int nsample;
    int nClasses = 6;
    cv::Mat trainData;
    cv::Mat trainLabel;
    
    void load_data(cv::String train_path) {
        int iter = 0;
        std::vector<std::string> v;
        std::vector<std::string> vfinal;
        std::vector<int> trainlab;
        // taking images name
        for (int lab = 0; lab < 6; lab++) {
            cv::String path(train_path + std::to_string(lab) + "/");
            std::vector<std::string> v2;
            read_directory(path, v2);
            v.insert(v.end(), v2.begin(), v2.end());
            // index vector
            for (size_t j = 0; j < v2.size(); j++) {
                trainlab.push_back(lab);
            }
        }
        std::vector<int> randid = randomvec(0, (int)v.size()-1, nsample);
        
        // rows = 100 ??? cols = 979104
        
        trainLabel = cv::Mat();
        
        for (size_t i = 0; i < nsample; i++) {
            int x = randid[i];
            vfinal.push_back(v[x]);
            trainLabel.push_back(trainlab[x]);
        }
        trainData = cv::Mat((int)vfinal.size(),979104,CV_32F);
        // converting in Mat
        for (auto &i : vfinal) {
            cv::Mat m = cv::Mat(task1(i)).t();
            // m.convertTo( m, CV_32F );
            m.copyTo(trainData.row(iter));
            //            trainData.row(iter).copyTo(m);
            
            iter++;
        }
        
    }
public:
    random_forest(int n, int samples, int mc, int f=0, int md=10, int ms=16) {
        cv::Ptr<cv::ml::DTrees> tree;
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
        nsample = samples;
        
//      load_data(train_path);
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
    
    // Needs urgent optimization!!
    void create_dataset(cv::String train_path)
    {
        trainData = cv::Mat(nsample, 800*608, CV_32F);
        int iter = 0;
        std::vector<std::string> v;
        std::vector<std::string> vfinal;
        std::vector<int> trainlab;
        // taking images name
        for (int lab = 0; lab < 6; lab++) {
            cv::String path(train_path + std::to_string(lab) + "/");
            std::vector<std::string> v2;
            read_directory(path, v2);
            v.insert(v.end(), v2.begin(), v2.end());
            // index vector
            for (size_t j = 0; j < v2.size(); j++) {
                trainlab.push_back(lab);
            }
        }
        std::vector<int> randid = randomvec(0, (int)v.size()-1, nsample);
        
        // rows = 100 ??? cols = 979104
        
        trainLabel = cv::Mat();
        
        for (size_t i = 0; i < nsample; i++) {
            int x = randid[i];
            vfinal.push_back(v[x]);
            trainLabel.push_back(trainlab[x]);
        }
        trainData = cv::Mat((int)vfinal.size(),979104,CV_32F);
        // converting in Mat
        for (auto &i : vfinal) {
            cv::Mat m = cv::Mat(task1(i)).t();
            // m.convertTo( m, CV_32F );
            m.copyTo(trainData.row(iter));
//            trainData.row(iter).copyTo(m);
            
            iter++;
        }
        
    }
    
    // putting the right training data and the train path can be chosen before (we use it multiple times) (we have to do it differently)
    void train(cv::String train_path){
        for (size_t i = 0; i < numTrees; i++)
        {
            create_dataset(train_path);
            dtrees[i]->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));
        }
    }
    
    int predict(cv::Mat sample) {
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
        return (int)std::distance(
                classes.begin(),
                std::max_element(classes.begin(),classes.end()));
    }
};

#endif
