#pragma once

#include <opencv2/opencv.hpp>

class ImgSegPrecisionMeasure{
public:
    ImgSegPrecisionMeasure(int numberOfClass);
    virtual ~ImgSegPrecisionMeasure();

    //return => the result scores. The return vector contains n+1 elements, where is the number os classes. The first n elements are the scores for each class, the last one is the average of all classes scores.
    virtual std::vector<double> measure(const cv::Mat& confusionMat) = 0;

    //predImg           => in image with type CV_32SC1 containing the predicted labels
    //gtImg             => in image with type CV_32SC1 containing the ground truth labels. Must have the same size of predImg.
    //inOutConfusionMat => inOut Mat with type CV_32SC1 containing the confusion matrix. The predicted labels are represented by columns, the predicted labels are represented by rows.
    virtual void fillConfusionMat(const cv::Mat& predImg, const cv::Mat& gtImg, cv::Mat& inOutConfusionMat);

protected:
    int numberOfClass;
};
