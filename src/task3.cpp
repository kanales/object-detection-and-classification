//
// Created by Iván Canales Martín on 2019-01-04.
//
#include "task3.h"

#include "RandomForest.hpp"
#include "ObjectDetector.h"
#include "DataLoader.h"

#include <fstream>


cv::String newPath( $ROOT "data/task3/train/0" );

void load_gt(std::vector<std::vector<DetectedObject>>& gts) {

    cv::String path( $ROOT "data/task3/gt/" );

    std::vector<std::string> v;
    v = read_directory(path);

    for (auto &nameimg : v) {

        cv::String index = nameimg;
        index = index.substr(index.size()-9);
        index = index.substr(0,index.size()-7);

        std::string line;
        std::ifstream myfile (nameimg);

        if (myfile.is_open())
        {
            std::vector<DetectedObject> v;

            while ( getline (myfile,line) )
            {
                //split
                std::istringstream buf(line);
                std::istream_iterator<std::string> beg(buf), end;
                std::vector<std::string> tokens(beg, end);

                DetectedObject obj {
                        std::stoi(tokens[0]),
                        cv::Rect(std::stoi(tokens[1]), std::stoi(tokens[2]), std::stoi(tokens[3])- std::stoi(tokens[1]), std::stoi(tokens[4])-std::stoi(tokens[2])),
                        1
                };

                v.push_back(obj);
            }
            myfile.close();
            //gts.push_back(v);
            gts[std::stoi(index)] = v;
            v.clear();
        }
    }
}


std::tuple<int, int, int> evaluate(std::vector<std::vector<DetectedObject>>& gts, int index, std::vector<DetectedObject>& objs, float thr, cv::Mat image){

    /*
    cv::String index = imagePath.substr(imagePath.size()-6);
    index = index.substr(0,index.size()-4);
    std::cout << std::stoi(index) << '\n';
    */

    int correct = 0;
    int falsepos = 0;
    int falseneg = 0;

/*
  cv::rectangle(image, gts[index][0].rect, cv::Scalar(6,57,113));
  cv::rectangle(image, gts[index][1].rect, cv::Scalar(6,57,113));
  cv::rectangle(image, gts[index][2].rect, cv::Scalar(6,57,113));
  cv::imshow("Image", image);
  cv::waitKey();
*/


    for(int i = 0; i< objs.size() ; i++){

          int cls = objs[i].cls;
          DetectedObject gt = gts[index][cls];

          // std::cout << ((float)(gt.rect & objs[i].rect).area() / (gt.rect | objs[i].rect).area()) << '\n';
          if( ((gt.rect & objs[i].rect).area() / (float)(gt.rect | objs[i].rect).area()) > thr) correct++;
          else falsepos++;
    }
    falseneg += 3-std::min(3,falsepos+correct);
    // not so sure


  // for(int i = 0; i< objs.size() ; i++){
  //
  //     int cls = objs[i].cls;
  //     DetectedObject gt = gts[index][cls];
  //
  //     // std::cout << ((float)(gt.rect & objs[i].rect).area() / (gt.rect | objs[i].rect).area()) << '\n';
  //     if( ((gt.rect & objs[i].rect).area() / (float)(gt.rect | objs[i].rect).area()) > thr) correct++;
  //     else if( ((gt.rect & objs[i].rect).area() / (float)(gt.rect | objs[i].rect).area()) < thr) falsepos++;
  //     else falseneg++;
  // }
  //
  //
  //   if (correct == 0 )
  //       std::cout << "bad, no object found" << '\n';
  //   else{
  //       std::cout << "Precision = " << (float)correct/(objs.size()+falsepos) << '\n';
  //       std::cout << "Recall = " << (float)correct/3*gts.size() << '\n';
  //   }
    return {correct,falsepos,falseneg};
}


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
    std::cout << "Augmenting data...";
    for (int lab = 0; lab < 4; lab++) {
        std::vector<std::string> v;
        cv::String string_lab(std::to_string(lab));
        cv::String path( train_path + string_lab + "/" );
        v = read_directory(path);
        for (auto &img : v) {
            image_flip(img,string_lab);
            for (int i = 0; i < 360; i+=90) {
                image_rotation(img,i,string_lab);
            }
        }
    }
}

cv::Scalar class_color[] = {cv::Scalar(0,0,255), cv::Scalar(0,255,0), cv::Scalar(255,0,0)};

// execute task 3
void part3(bool retrain, float object_thr, float overlapthr) {
    // cv::String path( $ROOT "data/task3/train/0" );
    cv::String path( $ROOT "data/task3/train/0" );
    cv::String test_path( $ROOT "data/task3/test/0000.jpg" );
    cv::String datatest_path( $ROOT "data/task3/test/000" );
    cv::String model_dir( $ROOT "model/" );
    // int imageIndex = 0;

    //std::cout << "Augmenting..." << std::endl;
    //data_augmentation(path);

    int n_classes = 4;
    int ntrees  = 50; //20

    int nsample = RandomForest::ALL_SAMPLES;

    cv::HOGDescriptor hog = mk_hog();
    RandomForest rf(ntrees, nsample, hog, n_classes, 25, 100, 0);

    DataLoader dl;


    if (retrain) {
        std::cout << "Training forest..." << std::endl;
        auto [feats, labels]  = dl.load(path,n_classes, hog); //rf.load_train(path);
        rf.train(feats, labels);
        rf.save(model_dir);
        std::cout << "Done training." << std::endl;
    } else {
        std::cout << "Loading forest..." << std::endl;
        rf.load(model_dir);
    }
    
    int precision = 0;
    int recall = 0;
    std::ofstream myfile;
    myfile.open ("data.dat");
    myfile << "#Recall Precision" << std::endl;
    for (object_thr ; object_thr < 1; object_thr+=0.9) {

    ObjectDetector od(rf, 3);
    od.overlap_thr = overlapthr;
    od.bgCutoff = object_thr;

    cv::Mat image = cv::imread(test_path, cv::IMREAD_COLOR);

    std::vector<DetectedObject> objs = od.detectObjects(image, 1.3, 8);

    for (auto el: objs) {
        std::cout << el.confidence << ' ';
    }

    std::ostringstream stream;
    for (DetectedObject v: objs) {
        cv::rectangle(image, v.rect, class_color[v.cls]);
        stream.str(std::string());
        stream << "cls: " << v.cls << " conf:" << v.confidence;
        cv::putText(image, stream.str(), cv::Point(v.rect.x, v.rect.y), cv::FONT_HERSHEY_PLAIN, 0.75, class_color[v.cls], 1);

    }

    std::cout << "Evaluating ..." << '\n';
    std::vector<std::vector<DetectedObject>> gts (44);
    load_gt(gts);

<<<<<<< HEAD
      int totalcorrects = 0;
      int totalfalsepos = 0;
      int totalfalseneg = 0;
      for (size_t img = 0; img < 3; img++) {
        std::cout << img << '\n';
        cv::String image_test_path( datatest_path+std::to_string(img)+".jpg");

        cv::Mat image_test = cv::imread(image_test_path, cv::IMREAD_COLOR);

        std::vector<DetectedObject> objs = od.detectObjects(image_test);
        //
        // for (auto obj: gts[0]) {
        //   cv::rectangle(image, obj.rect, cv::Scalar(0,0,0));
        // }
        //
        // cv::imshow("Image", image);
        // cv::waitKey();

        float thr = 0.5;
        auto [correct,falsepos,falseneg] = evaluate(gts, img, objs, thr, image);
        totalcorrects += correct;
        totalfalsepos += falsepos;
        totalfalseneg += falseneg;
      }
      if (totalcorrects == 0 )
          std::cout << "bad, no object found" << '\n';
      else{
          precision = (float)totalcorrects/(totalcorrects+totalfalsepos);
          recall = (float)totalcorrects/(totalcorrects+totalfalseneg);
          std::cout << "Thr = " << object_thr << '\n';
          std::cout << "Precision = " << precision << '\n';
          std::cout << "Recall = " << recall << '\n';
      }
      myfile << recall << " " << precision << std::endl;
=======
    for (auto obj: gts[imageIndex]) {
        cv::rectangle(image, obj.rect, cv::Scalar(0,0,0));
>>>>>>> bbbf2fef21a5e4ac7fb732f065fb37f76a7a8cf5
    }
    myfile.close();

    // TODO: change the thr and create the precision-recall chart

}
