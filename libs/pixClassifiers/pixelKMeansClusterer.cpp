// Copyright 2018

#include "pixelKMeansClusterer.hpp"
using namespace cv;

PixelKMeansClusterer::PixelKMeansClusterer(ImageFeatures* featureCalculator, uint k)
    : PixelClassifier(featureCalculator), k(k) {
}

PixelKMeansClusterer::~PixelKMeansClusterer() {
    
}

void PixelKMeansClusterer::setK(uint k) {
    this->k = k;
}
void PixelKMeansClusterer::setTermCriteria(TermCriteria criteria) {
    this->term = criteria;
}
void PixelKMeansClusterer::setNTries(uint tries) {
    this->tries = tries;
}
void PixelKMeansClusterer::setFlag(int flag) {
    assert(flag == KMEANS_PP_CENTERS);
    assert(flag == KMEANS_RANDOM_CENTERS);

    this->flag = flag;
}

void PixelKMeansClusterer::getCenters(Mat& outCenters) {
    outCenters = this->centers;
}

void PixelKMeansClusterer::classify(Mat& inImage, Mat& outLabelImage) {
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat featureImage, kmeansInput;
    this->featureCalculator->getFeatures(inImage, featureImage);
    cv::imwrite("feature_image.png", featureImage); ////////////////////////////////////////////////////
    kmeansInput = featureImage.reshape(1, inImage.total());

    assert(kmeansInput.rows == inImage.total());
    assert(kmeansInput.cols == this->featureCalculator->getDimentions());
    // TO-DO: fix get"Dimentions" name

    kmeans(kmeansInput,
           this->k,
           outLabelImage,
           this->term,
           this->tries,
           this->flag,
           this->centers);
}
