//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "config.h"
#include "utils.h"
#include "task1.h"
#include "task2.h"
#include "RandomForest.hpp"
#include "ObjectDetector.h"

int main(int argc, const char * argv[]) {
    part1(argc,argv);
    part2(argc, argv);
}
