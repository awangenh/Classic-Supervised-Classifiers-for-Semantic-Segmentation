#pragma once

#include "imageFeatures.hpp"
#include "imageFeatures/vegetationIndex/vegetationIndex.hpp"

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class RGBVegetationIndex : public ImageFeatures{
public:
    RGBVegetationIndex(VegetationIndex* vIndex);//Does not take the ownership!
    virtual ~RGBVegetationIndex();

    void getFeature(vector<Point2i> &pixCoordinates, vector<double *> &outFeatures);
    void getFeature(vector<double *> &outFeatures);

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures);
    void getFeatures(const Mat &image, Mat &outFeatureImage);

protected:
    VegetationIndex* vegetationIndex;
};
