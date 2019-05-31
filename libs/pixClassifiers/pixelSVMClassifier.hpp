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

class PixelSVMClassifier : public PixelClassifier{
public:
    enum Kernel{
        LINEAR=0,
        RBF=1,
    };
    PixelSVMClassifier(ImageFeatures* featureCalculator, Kernel k = LINEAR);
    virtual ~PixelSVMClassifier();

    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void setTrainData(Mat& imageSample, Mat& imageLabel);

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels);

    virtual void train();

    virtual void classify(Mat& inImage, Mat& outLabelImage);


protected:
    Ptr<ml::SVM> svm;

    Mat trainData;
    Mat trainDataLabels;
};
