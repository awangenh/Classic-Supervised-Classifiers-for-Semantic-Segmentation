#pragma once

#include "imageFeatures.hpp"
#include "gabor/gabor_texture.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class GaborFeature : public ImageFeatures{
public:
    GaborFeature(const GaborTexture& gabor, ImageFeatures* seedFeature);
    virtual ~GaborFeature();

    virtual void getFeature(vector<Point2i>& pixCoordinates, vector<double*>& outFeatures);
    virtual void getFeature(vector<double*>& outFeatures);

    virtual void getFeatures(const Mat& image, vector<Point2i>& pixCoordinates, Mat& outFeatures);
    virtual void getFeatures(const Mat& image, Mat& outFeatureImage);

private:
    GaborTexture gabor;
    ImageFeatures* seedFeature;
};
