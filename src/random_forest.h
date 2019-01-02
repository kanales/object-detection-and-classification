#include <opencv2/ml.hpp>

class random_forest{
private:
  std::vector<cv::ml::DTrees> dtree;
  int numTrees;
  int folds;
  int maxCategories;
  int maxDepth;
  int minSampleCount;
  int nsample;
  int nClasses = 6;
  cv::Mat trainData;
  cv::Mat trainLabel;
  cv::Mat testData;
public:
  random_forest(int n, int samples, int f, int mc, int md, int ms){
    dtree(n);
	for (int i = 0; i < n; i++)
	{
		dtree[i]->(cv::ml::DTrees::create());
    dtree[i]->setCVFolds(f /*10*/); // nonzero causes core dump
    dtree[i]->setMaxCategories(mc);
    dtree[i]->setMaxDepth(md);
    dtree[i]->setMinSampleCount(ms);
  }
  numTrees = n;
  nsample = samples;
  }

  void setCVFolds(int val){
    dtree->setCVFolds(val /*10*/); // nonzero causes core dump
  }

  void setMaxCategories(int val){
    dtree->setMaxCategories(val);
  }

  void setMaxDepth(int val){
    dtree->setMaxDepth(val);
  }

  void setSampleCount(int val){
    dtree->setMinSampleCount(val);
  }

  void create_test(std::vector<std::string> test_path)
  {
    int iter = 0;
    std::vector<std::string> v;
	  // test data
	  cv::String path2(testpath + "/02/");
	  read_directory(path2, v);
	  testData = (v.size(), 800*600, CV_32F);
	  iter = 0;
	  for (auto &i : v) {
		  std::cout << i << '\n';
		  cv::Mat m = cv::Mat(task1(i)).t();
		  // m.convertTo( m, CV_32F );
		  testData.row(iter).copyTo(m);
		  iter++;
	  }
  }

  void create_dataset(std::vector<std::string> train_path)
  {
	  trainData = (nsample, 800*600, CV_32F);
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
	  std::vector<int> randid = randomvec(0, v.size(), nsample);
	  for (size_t i = 0; i < nsample; i++) {
		  int x = randid[i];
		  vfinal.push_back(v[x]);
		  trainLabel.push_back(trainlab[x]);
	  }
	  // converting in Mat
	  for (auto &i : vfinal) {
		  std::cout << i << '\n';
		  cv::Mat m = cv::Mat(task1(i)).t();
		  // m.convertTo( m, CV_32F );
		  trainData.row(iter).copyTo(m);
		  iter++;
	  }

  }

  // putting the right training data and the train path can be choose before (we use it multiple times) (we have to do it differently)
  void train(std::vector<std::string> train_path){
  	for (size_t i = 0; i < numTrees; i++)
  	{
  		create_dataset(train_path);
  		dtree[i]->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));
  	}
  }

  int predict(cv::Mat test_descriptor) {
	  // create_test(test_path);
	  std::vector<int> preds;
	  std::vector<int> classes[nClasses];
	  for (size_t i = 0; i < numTrees; i++)
	  {
		  preds[i] = dtree[i]->predict(test_descriptor);
		  nClasses[preds[i]]++;
	  }
	  int maxPred = std::max_element(nClasses);
      std::cout << maxPred << '\n';
	  return maxPred;
  }
}
