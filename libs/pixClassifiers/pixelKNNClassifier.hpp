#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "pixelClassifier.hpp"

using namespace std;
using namespace cv;

class PixelKNNClassifier : public PixelClassifier{
public:
    PixelKNNClassifier(ImageFeatures* featureCalculator, uint k = 1);
    virtual ~PixelKNNClassifier();

    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void setTrainData(Mat& imageSample, Mat& imageLabel);

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels);

    virtual void train();

    virtual void classify(Mat& inImage, Mat& outLabelImage);

    void setK(uint k);

protected:
    Ptr<ml::KNearest> knn;
    uint k;

    Mat trainData;
    Mat trainDataLabels;
};
