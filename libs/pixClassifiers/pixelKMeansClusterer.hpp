// Copyright 2018

#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <memory>
#include <opencv2/opencv.hpp>
#include "pixelClassifier.hpp"

using namespace cv;

class PixelKMeansClusterer : public PixelClassifier {
 public:
    PixelKMeansClusterer(ImageFeatures* featureCalculator, uint k);
    virtual ~PixelKMeansClusterer();

    virtual void addTrainData(Mat& sampleImage, vector<vector<Point2i> >& samplePixCords) {}
    virtual void addTrainData(Mat& samples, Mat& labels) {}
    virtual void setTrainData(Mat& sampleImage,
                              vector<vector<Point2i>>& samplePixCords){}
    virtual void setTrainData(Mat& samples, Mat& labels) {}
    virtual void train() {}

    virtual void classify(Mat& inImage, Mat& outLabelImage);

    void setK(uint k);
    void setTermCriteria(TermCriteria criteria);
    void setNTries(uint tries);
    void setFlag(int flag);
    void getCenters(Mat& outCenters);

 protected:
    uint k;
    uint tries = 10;
    int  flag = KMEANS_PP_CENTERS;
    TermCriteria term = TermCriteria(TermCriteria::EPS, 10, 1.0);
    Mat centers;
};
