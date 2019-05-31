#pragma once

#include "imageFeatures.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class MultiFeature : public ImageFeatures{
public:
    MultiFeature();
    MultiFeature(vector<ImageFeatures*>& features);
    virtual ~MultiFeature();

    void concatFeat(ImageFeatures* f);

    void getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures);
    void getFeature(std::vector<double *> &outFeatures);

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures);
    void getFeatures(const Mat &image, Mat &outFeatureImage);

private:
    vector<ImageFeatures*> features;
};
