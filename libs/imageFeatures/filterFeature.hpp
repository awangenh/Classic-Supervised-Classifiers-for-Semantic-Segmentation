#pragma once

#include "imageFeatures.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class FilterFeature : public ImageFeatures{
public:
    FilterFeature(ImageFeatures* feature);
    virtual ~FilterFeature();

    void addFilter(Mat& filter);
    void addFilter(vector<Mat>& filters);

    void getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures);
    void getFeature(std::vector<double *> &outFeatures);

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures);
    void getFeatures(const Mat &image, Mat &outFeatureImage);

private:
    ImageFeatures* feature;
    vector<Mat> filters;
};
