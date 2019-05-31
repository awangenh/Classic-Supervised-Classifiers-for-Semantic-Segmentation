#pragma once

#include "imageFeatures.hpp"
#include "gabor/gabor_texture.hpp"
#include "imageFeatures/gaborFeatures.hpp"
#include <opencv2/opencv.hpp>

class RGBGaborFeature : public ImageFeatures{
public:
    RGBGaborFeature(const cv::Mat& image, const GaborTexture& gabor, bool preCalculated = true);
    virtual ~RGBGaborFeature();

    virtual void getFeature(std::vector<cv::Point2i>& pixCoordinates, std::vector<double*>& outFeatures);
    virtual void getFeature(std::vector<double*>& outFeatures);

    virtual void getFeatures(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outFeatures);
    virtual void getFeatures(const Mat& image, Mat& outFeatureImage);

private:
    std::vector<cv::Mat> data;
    cv::Mat image;
    cv::Size imageOriginalSize;
    int borderSize;
    const GaborTexture gabor;
    const bool preCalculated;
    GaborFeature* gaborFeature;
};
