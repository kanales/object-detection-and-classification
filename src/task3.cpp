//
// Created by Iván Canales Martín on 2019-01-04.
//
#include "task3.h"

#include "RandomForest.hpp"
#include "ObjectDetector.h"
#include "DataLoader.h"

cv::String newPath( $ROOT "data/task3/train_new/0" );

void image_rotation(cv::String imagePath, int angle, const cv::String &lab){

  cv::Mat src = cv::imread(imagePath); //, CV_LOAD_IMAGE_UNCHANGED);

  // get rotation matrix for rotating the image around its center in pixel coordinates
  cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle, center not relevant
  cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
  rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

  cv::Mat dst;
  cv::warpAffine(src, dst, rot, bbox.size());
  imagePath = imagePath.substr(0,imagePath.length()-4);
  cv::String imageName = imagePath.substr(imagePath.length()-4,imagePath.length());
  cv::imwrite(newPath + lab + "/" + imageName + lab + "rot" + std::to_string(angle) + ".jpg", dst);   // put a different name

}

void image_flip(cv::String imagePath, const cv::String &lab){

  // do this for all the image??
  cv::Mat src = cv::imread(imagePath);
  cv::Mat dst;               // dst must be a different Mat
  cv::flip(src, dst, 1);     // because you can't flip in-place (leads to segfault)
  imagePath = imagePath.substr(0,imagePath.length()-4);
  cv::String imageName = imagePath.substr(imagePath.length()-4,imagePath.length());
  cv::imwrite(newPath + lab + "/" + imageName + lab + "flip.jpg", dst);   // put a different name
}

void data_augmentation(cv::String train_path) {
  for (int lab = 0; lab < 4; lab++) {
    std::vector<std::string> v;
    cv::String string_lab(std::to_string(lab));
    cv::String path( train_path + string_lab + "/" );
    read_directory(path, v);
    for (auto &img : v) {
        image_flip(img,string_lab);
        for (int i = 0; i < 360; i+=90) {
          image_rotation(img,i,string_lab);
        }
    }
  }
}




// execute task 3
void part3(int argc, const char *argv[]) {
  // cv::String path( $ROOT "data/task3/train/0" );
  cv::String path( $ROOT "data/task3/train/0" );
  cv::String path2( $ROOT "data/task3/test/0000.jpg" );

  // std::cout << "Augmenting..." << std::endl;
  // data_augmentation(path);

  int n_classes = 4;
  int ntrees  = 20;
  int nsample = 500;//RandomForest::ALL_SAMPLES;

  cv::HOGDescriptor hog = mk_hog();
  RandomForest rf(ntrees,nsample, hog, n_classes);

  DataLoader dl;
  std::cout << "Training forest..." << std::endl;

  auto [feats, labels]  = dl.load(path,n_classes, hog); //rf.load_train(path);
  rf.train(feats, labels);
  std::cout << "Done training." << std::endl;

  ObjectDetector od(rf, 3);

  cv::Mat image = cv::imread(path2, cv::IMREAD_COLOR);

  std::vector<DetectedObject> objs = od.detectObjects(image);

  for (auto el: objs) {
    std::cout << el.confidence << ' ';
  }

  for (DetectedObject v: objs) {
    cv::rectangle(image, v.rect, cv::Scalar(0,0,255));

  }
  cv::imshow("Image", image);
  cv::waitKey();
}
