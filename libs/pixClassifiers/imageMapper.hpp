#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <memory>

#include "pixelClassifier.hpp"

using namespace std;
using namespace cv;

class ImageMapper{
public:
    ImageMapper(PixelClassifier* classifier);//takes the ownership
    virtual ~ImageMapper();

    virtual void doMapping(Mat& inImage, Mat& outMap);

protected:
    PixelClassifier* classifier;
};
