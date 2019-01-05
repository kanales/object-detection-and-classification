//
// Created by Iván Canales Martín on 2019-01-04.
//

#include "task2.h"

std::vector<cv::Mat> load_test(cv::String test_path, char val) {
    cv::Mat image;
    std::vector<std::string> v;
    cv::String path2(test_path + "0" + val + "/");
    read_directory(path2, v);
    std::vector<cv::Mat> out(v.size());
    int idx = 0;
    for (auto &s: v) {
        out[idx++] = cv::imread(s, cv::IMREAD_COLOR);
    }
    return out;
}

// execute task 2
void part2(int argc, const char *argv[]) {
    cv::String path( $ROOT "data/task2/train/0" );

    cv::String path2( $ROOT "data/task2/test/" );
    //cv::String path2( $ROOT "data/task2/train/" );


    int ntrees  = 20;
    int nsample = 500;//RandomForest::ALL_SAMPLES;

    cv::HOGDescriptor hog = mk_hog();
    RandomForest rf(ntrees,nsample, hog, 6);

    std::cout << "Training forest..." << std::endl;
    auto [feats, labels]  = rf.load_train(path);
    rf.train(feats, labels);
    std::cout << "Done training." << std::endl;

    std::cout << "Predicting..." << std::endl;

    char values[6] = {'0','1','2','3','4','5'};
    int correct = 0, total = 0;
    for (Class j = 0; j < 6; j++) {
        std::cout << "Expected " << values[j] << ": " << std::endl;
        std::vector<cv::Mat> images = load_test(path2, values[j]);
        for (auto img: images) {
            std::vector<float> pred = rf.predictImage(img);
            int k = (int)std::distance(pred.begin(), std::max_element(pred.begin(), pred.end()));
            std::cout << "\tP: " << k << ' ';
            print_vector(pred);
            std::cout << std::endl;
            total++;
            if (j == k) correct++;
        }
    }
    std::cout << "Accuracy: " << ((float)correct) / total << '%' << std::endl;
}