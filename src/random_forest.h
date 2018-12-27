#include <opencv2/ml.hpp>

class random_forest{
private:
  cv::DTrees dtree;
  int numTrees;
  int folds;
  int maxCategories;
  int maxDepth;
  int minSampleCount;
public:
  random_forest(int n, int f, int mc, int md, int ms){
    dtree = cv::DTrees.create();
    dtree->setCVFolds(f /*10*/); // nonzero causes core dump
    dtree->setMaxCategories(mc);
    dtree->setMaxDepth(md);
    dtree->setMinSampleCount(ms);
  }

  void setCVFolds(int val){
    dtree->setCVFolds(f /*10*/); // nonzero causes core dump
  }

  void setMaxCategories(int val){
    dtree->setMaxCategories(mc);
  }

  void setMaxDepth(int val){
    dtree->setMaxDepth(md);
  }

  void setSampleCount(int val){
    dtree->setMinSampleCount(ms);
  }

  void train(cv::Mat trainData, cv::Mat trainLabel){
    dtree->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));
  }

  void predict(cv::Mat testData) {
        int preds = dtree->predict(testData);
        std::cout << preds << '\n';
  }
}
