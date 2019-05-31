#pragma once

#include <opencv2/opencv.hpp>

#include "imgSegPrecisionMeasure.hpp"

class ImgF1Score : public ImgSegPrecisionMeasure{
public:
    ImgF1Score(int numberOfClass);
    virtual ~ImgF1Score();
    std::vector<double> measure(const cv::Mat& confusionMat);
private:

};
