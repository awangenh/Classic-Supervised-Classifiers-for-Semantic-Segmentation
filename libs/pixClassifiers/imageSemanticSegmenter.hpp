#pragma once

#include <iostream>
#include <vector>
#include <set>
#include <opencv2/opencv.hpp>
#include <memory>

#include "pixelClassifier.hpp"

using namespace std;
using namespace cv;

class ImageSemanticSegmenter{
public:
    ImageSemanticSegmenter(PixelClassifier* classifier);//takes the ownership
    virtual ~ImageSemanticSegmenter();

    virtual void doSegmentation(Mat& inImage, Mat& outMap);

protected:
    PixelClassifier* classifier;
};
