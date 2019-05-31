#include "filterFeature.hpp"

using namespace std;
using namespace cv;

FilterFeature::FilterFeature(ImageFeatures *feature)
    : feature(feature)
{
    dimensions = feature->getDimentions();
}

FilterFeature::~FilterFeature()
{

}

void FilterFeature::addFilter(Mat &filter)
{
    filters.push_back(filter);
    dimensions = feature->getDimentions() * filters.size();
}

void FilterFeature::addFilter(vector<Mat> &filters)
{
    for(Mat& filter : filters)
        this->filters.push_back(filter);
    dimensions = feature->getDimentions() * filters.size();
}

void FilterFeature::getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures)
{

}

void FilterFeature::getFeature(std::vector<double *> &outFeatures)
{

}

void FilterFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
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

    vector<Mat> featuresMats;
    Mat filtered;
    for(Mat& filter : filters){
        Mat outFeature;
        cv::filter2D(image, filtered, CV_32F, filter, Point(-1,-1), 0, BORDER_REFLECT);
        filtered.convertTo(filtered, CV_8U);
        feature->getFeatures(filtered, pixCoordinates, outFeature);
        assert(outFeature.rows == pixCoordinates.size());
        assert(outFeature.cols == feature->getDimentions());
        featuresMats.push_back(outFeature);
    }
    cv::hconcat(featuresMats, outFeatures);
    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
}

void FilterFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
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

    vector<Mat> featuresMats;
    Mat filtered;
    for(Mat& filter : filters){
        Mat outFeature;
        cv::filter2D(image, filtered, CV_32F, filter, Point(-1,-1), 0, BORDER_REFLECT);
        imwrite("filtered.png", filtered);
        filtered.convertTo(filtered, CV_8U);
        feature->getFeatures(filtered, outFeature);
        outFeature = outFeature.reshape(1, image.total());
        assert(outFeature.cols == feature->getDimentions());
        featuresMats.push_back(outFeature);
    }
    cv::hconcat(featuresMats, outFeatureImage);
    assert(outFeatureImage.rows = image.total());
    assert(outFeatureImage.cols = dimensions);
    outFeatureImage = outFeatureImage.reshape(1, image.rows);
    assert(outFeatureImage.rows = image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
}
