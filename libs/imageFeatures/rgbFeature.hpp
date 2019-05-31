#pragma once

#include "imageFeatures.hpp"
#include <opencv2/opencv.hpp>

class RGBFeature : public ImageFeatures{
public:
    RGBFeature();
    virtual ~RGBFeature();

    void getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures);
    void getFeature(std::vector<double *> &outFeatures);

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures);
    void getFeatures(const Mat &image, Mat &outFeatureImage);

private:

};
