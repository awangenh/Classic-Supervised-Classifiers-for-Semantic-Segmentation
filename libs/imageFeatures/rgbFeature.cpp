#include "rgbFeature.hpp"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

RGBFeature::RGBFeature()
{
    dimensions = 3;
}

RGBFeature::~RGBFeature()
{

}

void RGBFeature::getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures)
{

}

void RGBFeature::getFeature(std::vector<double *> &outFeatures)
{

}

void RGBFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
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

    for(int i = 0; i < pixCoordinates.size(); i++){
        Point2i coord = pixCoordinates[i];
        Vec3b bgr = image.at<Vec3b>(coord);
        Vec3f& feature = outFeatures.at<Vec3f>(i, 0);
        feature[0] = bgr[0] * (2.0/255.0) + (-1.0);
        feature[1] = bgr[1] * (2.0/255.0) + (-1.0);
        feature[2] = bgr[2] * (2.0/255.0) + (-1.0);
    }

}

void RGBFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
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

    Mat imageTemp = image.reshape(1, 0);
    imageTemp.convertTo(outFeatureImage, CV_32F, (2.0/255.0), -1.0);
}

