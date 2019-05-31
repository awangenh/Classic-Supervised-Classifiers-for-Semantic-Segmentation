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

class IB2Mahala{
public:
    IB2Mahala(uint order = 2);
    virtual ~IB2Mahala();

    virtual void train(Mat& samples, Mat& labels);
    virtual void classify(Mat& data, Mat& outLabels);

protected:
    typedef classifiers::polyMahalanobis PolyMahala;

    Mat trainData;
    Mat trainLabels;

    Mat cd;//concept descriptor.
    Mat cdLabels;//label of each cd element.

    PolyMahala* mahala;
    int order;
    vector<int> labelsNumbers;
};
