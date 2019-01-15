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

int main(int argc, const char * argv[]) {
    /*
    for (int i = 0; i < 3; i++) {
        part1(i);
    }

    float max = 0;
    int maxi = 0;
    for (int i = 10; i < 100; i += 5) {
        float acc = part2(i);
        std::cout << i << " > " << acc << std::endl;
        if (acc > max) {
            max = acc;
            maxi = i;
        }
    }



    std::cout << maxi << ": acc " << max << std::endl;
     */
    ///*
    //part1(0);
    part3(false, 0.7, 0.75);
    //*/
}
