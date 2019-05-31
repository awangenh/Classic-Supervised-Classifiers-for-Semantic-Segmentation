#pragma once

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class ImageFeatures{
public:

    uint getDimentions() const;
    virtual void getFeature(std::vector<cv::Point2i>& pixCoordinates, std::vector<double*>& outFeatures) = 0;
    virtual void getFeature(std::vector<double*>& outFeatures) = 0;

    virtual void getFeatures(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outFeatures) = 0;
    virtual void getFeatures(const Mat& image, Mat& outFeatureImage) = 0;

protected:

    uint dimensions;

};
