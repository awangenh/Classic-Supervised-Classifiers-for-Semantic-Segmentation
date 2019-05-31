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

class PixelRandomForestClassifier : public PixelClassifier{
public:
    PixelRandomForestClassifier(ImageFeatures* featureCalculator, uint n_tree = 50, uint max_deep = 30, uint max_categorie = 16);
    virtual ~PixelRandomForestClassifier();

    virtual void setTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void setTrainData(Mat& imageSample, Mat& imageLabel);

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords);
    virtual void addTrainData(Mat& imageSamples, Mat& imageLabels);

    virtual void train();

    virtual void classify(Mat& inImage, Mat& outLabelImage);


protected:
    Ptr<ml::RTrees> rf;

    Mat trainData;
    Mat trainDataLabels;
};
