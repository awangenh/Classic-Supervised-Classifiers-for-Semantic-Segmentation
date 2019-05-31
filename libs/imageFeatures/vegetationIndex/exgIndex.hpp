#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "imageFeatures/vegetationIndex/vegetationIndex.hpp"

using namespace std;
using namespace cv;

class ExGIndex : public VegetationIndex{
public:
    ExGIndex();

    virtual void getIndex(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outIndexes);
    virtual void getIndex(const Mat& image, Mat& outIndexImage);

protected:

};
