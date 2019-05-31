#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <memory>

#include "imageMapper.hpp"

using namespace std;
using namespace cv;

class WeedMapper : public ImageMapper{
public:
    WeedMapper(PixelClassifier* classifier);//does not take the ownership
    virtual ~WeedMapper();

    virtual void setTrainData(Mat& sampleImage, vector<Point2i>& plantSamplePixCoords, vector<Point2i>& weedSamplePixCoords, vector<Point2i>& soilSamplePixCoords);
    virtual void doWeedMapping(Mat& inImage, Mat& outWeedMap);

private:

};
