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

class MahalanobisClassifier{
public:
    MahalanobisClassifier(uint order = 2);
    virtual ~MahalanobisClassifier();

    virtual void train(Mat& samples, Mat& labels);
    virtual void classify(Mat& data, Mat& outLabels);

protected:
    typedef classifiers::polyMahalanobis PolyMahala;

    vector<PolyMahala*> mahalas;
    vector<int> labelsNumbers;
    int order;
};
