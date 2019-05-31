#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>

#include "pixelClassifier.hpp"
#include "mahalanobisClassifier.hpp"
#include "mahalanobisKNN.hpp"
#include "ib2Mahala.hpp"

using namespace std;
using namespace cv;

class PixelMahalanobisClassifier : public PixelClassifier{
public:
    PixelMahalanobisClassifier(ImageFeatures* featureCalculator);
    PixelMahalanobisClassifier(ImageFeatures* featureCalculator, uint order);
    virtual ~PixelMahalanobisClassifier();

    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void setTrainData(Mat& imageSample, Mat& imageLabel);

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels);

    virtual void train();

    virtual void classify(Mat& inImage, Mat& outLabelImage);

protected:
    MahalanobisClassifier mahalaClassifier;
//    MahalanobisKNN mahalaClassifier;
//    IB2Mahala mahalaClassifier;

    Mat trainData;
    Mat trainDataLabels;
};
