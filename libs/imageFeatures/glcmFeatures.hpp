#pragma once

#include <opencv2/opencv.hpp>
#include "imageFeatures.hpp"
#include "glcm.hpp"

using namespace std;
using namespace cv;

class GLCMFeature : public ImageFeatures{
public:
    enum Feats{Contrast = 1U, Dissimilarity = 2U, Homogeneity = 4U,
               Energy = 8U, Entropy = 16U, Mean = 32U, StdDev = 64U,
               Correlation = 128U,
               ALL = 255U};
    GLCMFeature(vector<GLCM> glcms = {GLCM()}, uint windowSize = 9, uint feats = Contrast|Energy|Entropy|Correlation);
    virtual ~GLCMFeature();

    void getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures);
    void getFeature(std::vector<double *> &outFeatures);

    void getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures);
    void getFeatures(const Mat &image, Mat &outFeatureImage);

    void setUseFeat(uint feat);
    void setDontUseFeat(uint feat);
    bool useFeat(uint feat);


private:
    std::vector<float> computeFeatures(const Mat& p);
    float computeContrast(const Mat& p);
    float computeDissimilarity(const Mat& p);
    float computeHomogeneity(const Mat& p);
    float computeEnergy(const Mat& p);
    float computeEntropy(const Mat& p);
    float computeMean(const Mat& p, float *outMeanI = 0, float *outMeanJ = 0);
    float computeStdDev(const Mat& p, float meanI, float meanJ, float *outVarI = 0, float *outVarJ = 0);
    float computeCorrelation(const Mat& p, float meanI, float meanJ, float varI, float varJ);

    vector<GLCM> glcms;
    uint windowSize;
    uint featuresToUse;
};
