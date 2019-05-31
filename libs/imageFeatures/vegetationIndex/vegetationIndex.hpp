#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

#include "imageFeatures/imageFeatures.hpp"

using namespace std;
using namespace cv;

class VegetationIndex : public ImageFeatures{
public:

    virtual void getFeature(std::vector<cv::Point2i>& pixCoordinates, std::vector<double*>& outFeatures);
    virtual void getFeature(std::vector<double*>& outFeatures);

    virtual void getFeatures(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outFeatures);
    virtual void getFeatures(const Mat& image, Mat& outFeatureImage);

    virtual void getIndex(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outIndexes) = 0;
    virtual void getIndex(const Mat& image, Mat& outIndexImage) = 0;

protected:

};
