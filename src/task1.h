#include <opencv2/opencv.hpp>

#include "config.h"

#define CELL_SIZE 16
#define NBINS 8

std::vector<float> task1(cv::String imageName){
  cv::Mat image, editedImage, grayImg;
  std::vector<float> descriptor;
  image = cv::imread(imageName, cv::IMREAD_COLOR);

  if( image.empty() )                      // Check for invalid input
  {
      std::cout <<  "Could not open or find the image " << imageName << std::endl ;
  }
  else{

    cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
    cv::resize(image, editedImage, cv::Size(800,608)); // Check later if it's correct!
    cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

    // HOG descriptor
    cv::HOGDescriptor hog;
    hog.blockSize = cv::Size(2*CELL_SIZE,2*CELL_SIZE);
    hog.blockStride = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.cellSize = cv::Size(CELL_SIZE,CELL_SIZE);
    hog.nbins = NBINS;

    hog.compute(grayImg,descriptor);

    // for task 1 execute this
    
    // descriptor = task1(imageName);
    // visualizeHOG(grayImg, descriptor, hog);
    // cv::imwrite( $ROOT "output/test.jpg", grayImg);
    // cv::waitKey(0);

  }
  return descriptor;
}
