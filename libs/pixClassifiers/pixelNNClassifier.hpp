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

class PixelNNClassifier : public PixelClassifier{
public:
    PixelNNClassifier(ImageFeatures* featureCalculator);
    virtual ~PixelNNClassifier();

    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void setTrainData(Mat& imageSample, Mat& imageLabel);

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels);

    virtual void train();

    virtual void classify(Mat& inImage, Mat& outLabelImage);

protected:
    virtual Mat toOneHot() const;
    virtual Mat fromOneHot(const Mat& oneHotLabels) const;

    Ptr<ml::ANN_MLP> nn;

    std::set<int> classesLabels;

    Mat trainData;
    Mat trainDataLabels;
};
