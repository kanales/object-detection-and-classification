//
//  hog_visualization.h
//  object-detection-and-classification
//
//  Created by Iván Canales Martín on 18/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//

#ifndef hog_visualization_h
#define hog_visualization_h

void visualizeHOG(cv::Mat img, std::vector<float> &feats, cv::HOGDescriptor hog_detector, int scale_factor = 3);

#endif /* hog_visualization_h */
