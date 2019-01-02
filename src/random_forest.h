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
  cv::Mat testData;
public:
  random_forest(int n, int samples, int f, int mc, int md, int ms){
    numTrees = n;
    dtrees.reserve(numTrees);
    for (int i = 0; i < numTrees; i++)
  	{
  		dtrees[i]->cv::ml::DTrees::create();
      dtrees[i]->setCVFolds(f /*10*/);
      dtrees[i]->setMaxCategories(mc);
      dtrees[i]->setMaxDepth(md);
      dtrees[i]->setMinSampleCount(ms);
    }
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

  cv::Mat create_test(cv::String test_path)
  {
    int iter = 0;
    std::vector<std::string> v;
	  // test data
	  cv::String path2(test_path + "/02/");
	  read_directory(path2, v);
	  testData = (v.size(), 800*608, CV_32F);
	  iter = 0;
	  for (auto &i : v) {
		  std::cout << i << '\n';
		  cv::Mat m = cv::Mat(task1(i)).t();
		  // m.convertTo( m, CV_32F );
		  testData.row(iter).copyTo(m);
		  iter++;
	  }
    return testData;
  }

  void create_dataset(cv::String train_path)
  {
	  trainData = (nsample, 800*608, CV_32F);
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
  void train(cv::String train_path){
  	for (size_t i = 0; i < numTrees; i++)
  	{
  		create_dataset(train_path);
  		dtrees[i]->train(cv::ml::TrainData::create(trainData, cv::ml::ROW_SAMPLE, trainLabel));
  	}
  }

  int predict(cv::Mat test_descriptor) {
	  // create_test(test_path);
    std::vector<int> preds;
	  std::vector<int> classes(nClasses);
	  for (size_t i = 0; i < numTrees; i++)
	  {
		  preds[i] = dtrees[i]->predict(test_descriptor);
		  classes[preds[i]]++;
	  }
	  int maxPred = std::distance(classes.begin(),std::max_element(classes.begin(),classes.end()));
    std::cout << maxPred << '\n';
	  return maxPred;
  }
};
