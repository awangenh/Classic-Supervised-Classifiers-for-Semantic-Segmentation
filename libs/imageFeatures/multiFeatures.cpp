#include "multiFeatures.hpp"

using namespace std;
using namespace cv;


MultiFeature::MultiFeature()
{
    dimensions = 0;
}

MultiFeature::MultiFeature(vector<ImageFeatures *> &features)
{
    this->features = features;
    dimensions = 0;
    for(ImageFeatures* f : this->features){
        dimensions += f->getDimentions();
    }
}

MultiFeature::~MultiFeature()
{

}

void MultiFeature::concatFeat(ImageFeatures *f)
{
    features.push_back(f);
    dimensions += f->getDimentions();
}

void MultiFeature::getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures)
{

}

void MultiFeature::getFeature(std::vector<double *> &outFeatures)
{

}

void MultiFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
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
    for(ImageFeatures* f : features){
        Mat outF;
        f->getFeatures(image, pixCoordinates, outF);
        assert(outF.rows == pixCoordinates.size());
        assert(outF.cols == f->getDimentions());
        featuresMats.push_back(outF);
    }
    cv::hconcat(featuresMats, outFeatures);
    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
}

void MultiFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
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
    for(ImageFeatures* f : features){
        Mat outF;
        f->getFeatures(image, outF);
        outF = outF.reshape(1, image.total());
        assert(outF.cols == f->getDimentions());
        featuresMats.push_back(outF);
    }
    cv::hconcat(featuresMats, outFeatureImage);
    assert(outFeatureImage.rows = image.total());
    assert(outFeatureImage.cols = dimensions);
    outFeatureImage = outFeatureImage.reshape(1, image.rows);
    assert(outFeatureImage.rows = image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
}
