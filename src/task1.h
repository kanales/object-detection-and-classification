#include <opencv2/opencv.hpp>

#include "config.h"

std::vector<float> task1(cv::String imageName){
  cv::Mat image, editedImage, grayImg;
  std::vector<float> descriptor;
  image = cv::imread(imageName, cv::IMREAD_COLOR);

  if( image.empty() )                      // Check for invalid input
  {
      std::cout <<  "Could not open or find the image " << imageName << std::endl ;
  }
  else{
    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE ); // Create a window for display.

    // cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE);
    cv::copyMakeBorder(image, editedImage, 3, 4, 1, 0, cv::BORDER_REFLECT);
    // std::cout << cv::Size(image.cols * 5, image.rows * 5) << '\n';
    // return descriptor;
    resize(image, editedImage, cv::Size(800,600));
    // cv::imshow("Display window", editedImage);
    //
    cv::cvtColor(editedImage, grayImg, cv::COLOR_RGB2GRAY);

    // HOG descriptor
    cv::HOGDescriptor hog(cv::Size(64, 128), cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9);
    // hog.winSize = grayImg.size();
    //    std::vector<cv::Point> positions;
    //    positions.push_back(cv::Point(grayImg.cols / 2, grayImg.rows / 2));

    hog.compute(grayImg,descriptor);
  }
  return descriptor;
}
