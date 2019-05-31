#pragma once

#include <opencv2/opencv.hpp>

#include "imgSegPrecisionMeasure.hpp"

class ImgIOU : public ImgSegPrecisionMeasure{
public:
    ImgIOU(int numberOfClass);
    virtual ~ImgIOU();
    std::vector<double> measure(const cv::Mat& confusionMat);
private:

};
