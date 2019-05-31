#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "imageFeatures/vegetationIndex/vegetationIndex.hpp"

using namespace std;
using namespace cv;

class ExGExRIndex : public VegetationIndex{
public:
    enum{Subtraction = 0, biDimesion = 1};
    ExGExRIndex(int method = 1);

    virtual void getIndex(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outIndexes);
    virtual void getIndex(const Mat& image, Mat& outIndexImage);

protected:
    int method;
};
