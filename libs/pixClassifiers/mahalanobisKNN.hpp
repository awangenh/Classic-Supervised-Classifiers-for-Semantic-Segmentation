#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <memory>

#include <opencv2/opencv.hpp>

#include "mahalanobis/polynomialMahalanobis.h"
#include "mahalanobis/pattern.h"

using namespace std;
using namespace cv;

class MahalanobisKNN{
public:
    MahalanobisKNN(uint order = 2);
    virtual ~MahalanobisKNN();

    virtual void train(Mat& samples, Mat& labels);
    virtual void classify(Mat& data, Mat& outLabels);

protected:
    typedef classifiers::polyMahalanobis PolyMahala;

    Mat trainData;
    Mat trainLabels;

    PolyMahala* mahala;
    vector<int> labelsNumbers;
    int order;
};
