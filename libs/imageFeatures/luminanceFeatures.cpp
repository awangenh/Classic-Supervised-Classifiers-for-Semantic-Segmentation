#include "luminanceFeatures.hpp"
#include "utils/util.hpp"
#include "debug.h"
#include <omp.h>

using namespace std;
using namespace cv;

LuminanceFeature::LuminanceFeature()
    : ImageFeatures()
{
    dimensions = 1;
}

LuminanceFeature::~LuminanceFeature()
{

}

void LuminanceFeature::getFeature(vector<Point2i> &pixCoordinates, vector<double *> &outFeatures)
{

}

void LuminanceFeature::getFeature(vector<double *> &outFeatures)
{

}

void LuminanceFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatures.data == NULL){
        outFeatures = Mat(pixCoordinates.size(), dimensions, CV_32FC1);
    }
    else{
        assert(outFeatures.rows == pixCoordinates.size());
        assert(outFeatures.cols == dimensions);
        assert(outFeatures.type() == CV_32FC1);
    }
    Mat hslMat;
    cvtColor(image, hslMat, CV_BGR2HLS);
    for(int i = 0; i < pixCoordinates.size(); i++){
        const Point2i& coord = pixCoordinates[i];
        const Vec3b& hsl = hslMat.at<Vec3b>(coord);
        outFeatures.at<float>(i, 0) = (hsl[1]*2)/255.0 -1.0;
    }

    if(DEBUG){
        double min, max;
        minMaxLoc(outFeatures, &min, &max);
        assert(min >= -1 && max <= 1);
    }

    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
}

void LuminanceFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatureImage.data == NULL){
        outFeatureImage = Mat(image.rows, image.cols*dimensions, CV_32FC1);
    }
    else{
        assert(outFeatureImage.rows == image.rows);
        assert(outFeatureImage.cols == image.cols*dimensions);
        assert(outFeatureImage.type() == CV_32FC1);
    }

    Mat hsl;
    cvtColor(image, hsl, CV_BGR2HLS);
    vector<Mat> channels;
    cv::split(hsl, channels);
    channels[1].convertTo(outFeatureImage, CV_32F, 2.0/255.0, -1.0);

    if(DEBUG){
        double min, max;
        minMaxLoc(outFeatureImage, &min, &max);
        assert(min >= -1 && max <= 1);
    }

    assert(outFeatureImage.rows = image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
}


