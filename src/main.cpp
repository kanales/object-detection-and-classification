//
//  main.cpp
//  Object detection and classification
//
//  Created by Iván Canales Martín on 14/12/2018.
//  Copyright © 2018 Iván Canales Martín. All rights reserved.
//

#include "task1.h"
#include "task2.h"
#include "task3.h"

bool require_training() {
    bool x;
    std::cout << "Should the model be re-trained?" << std::endl;
    std::cin >> x;
    return x;
}

int get_index() {
    int x;
    std::cout << "Insert image index:" << std::endl;
    std::cin >> x;
    return x;
}


float get_bgcutoff() {
    float x;
    std::cout << "Insert background discrimination threshold:" << std::endl;
    std::cin >> x;
    return x;
}

float get_overlapthr() {
    float x;
    std::cout << "Insert maximum supression threshold:" << std::endl;
    std::cin >> x;
    return x;
}

int main(int argc, const char * argv[]) {
    //part1(0);
    //part2(25);
    // Select true to rebuild the forest, false otherwise
    int part;
    while (true) {
        std::cout << "Insert part:" << std::endl;
        std::cin >> part;
        switch (part) {
            case 1: part1(0);break;
            case 2: part2(25);break;
            case 31:
                test(require_training(), get_index(), get_overlapthr(), get_bgcutoff());break;
            case 32: part3(require_training(), 0, 0.75);break;
            case -1: return 0;
        }
    }
}
