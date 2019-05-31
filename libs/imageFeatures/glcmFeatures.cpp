#include "glcmFeatures.hpp"
#include "utils/util.hpp"
#include "debug.h"
#include <math.h>
#include <bitset>
#include <omp.h>

using namespace std;
using namespace cv;

GLCMFeature::GLCMFeature(vector<GLCM> glcms, uint windowSize, uint feats)
    : glcms(std::move(glcms)), windowSize(windowSize), featuresToUse(feats)
{
    assert(windowSize%2 == 1);
    dimensions = std::bitset<sizeof(uint)*8>(featuresToUse).count() * 3 * this->glcms.size();
}

GLCMFeature::~GLCMFeature()
{

}

void GLCMFeature::getFeature(std::vector<Point2i> &pixCoordinates, std::vector<double *> &outFeatures)
{

}

void GLCMFeature::getFeature(std::vector<double *> &outFeatures)
{

}

void GLCMFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatures.data == NULL){
        outFeatures = Mat(pixCoordinates.size(), dimensions, CV_32FC1);
    }
    else{
        assert(outFeatures.rows == pixCoordinates.size());
        assert(outFeatures.cols == dimensions);
        assert(outFeatures.type() == CV_32FC1);
    }
    static const int numThread = omp_get_max_threads();

    Mat borderImage;
    cv::copyMakeBorder(image, borderImage, windowSize/2, windowSize/2, windowSize/2, windowSize/2, BORDER_REFLECT);
    vector<Mat> channels;
    cv::split(borderImage, channels);
    for(int c = 0; c < 3; c++){
        const Mat& plane = channels[c];
        #pragma omp parallel for num_threads(numThread)
        for(int i = 0; i < pixCoordinates.size(); i++){
            const Point& pi = pixCoordinates[i];
            const Mat roi = plane(Rect(pi, Size(windowSize,windowSize)));
            float* outFeatureRowPtr = outFeatures.ptr<float>(i);
            Mat glcmMat;
            for(int g = 0; g < glcms.size(); g++){
                const GLCM& glcm = glcms[g];
                glcm.getGLCMMat(roi, glcmMat);
                vector<float> feats = computeFeatures(glcmMat);
                assert(feats.size() == dimensions/(3*glcms.size()));
                for(int d = 0; d < feats.size(); d++){
                    int rowIndex = c*glcms.size()*feats.size() + g*feats.size() + d;
                    assert(rowIndex >= 0 && rowIndex < outFeatures.cols);
                    outFeatureRowPtr[rowIndex] = feats[d];
                }
            }
        }
    }
}

void GLCMFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outFeatureImage.data == NULL){
        outFeatureImage = Mat(image.rows, image.cols*dimensions, CV_32FC1);
    }
    else{
        assert(outFeatureImage.rows == image.rows);
        assert(outFeatureImage.cols == image.cols*dimensions);
        assert(outFeatureImage.type() == CV_32FC1);
    }
    static const int numThread = omp_get_max_threads();

    PRINT_DEBUG("GLCM GET FEATURES START!");

    Mat borderImage;
    cv::copyMakeBorder(image, borderImage, windowSize/2, windowSize/2, windowSize/2, windowSize/2, BORDER_REFLECT);
    vector<Mat> channels;
    cv::split(borderImage, channels);
    for(int c = 0; c < 3; c++){
        const Mat& plane = channels[c];
        #pragma omp parallel for num_threads(numThread) collapse(2)
        for(int i = 0; i < image.rows; i++){
            for(int j = 0; j < image.cols; j++){
                const Mat roi = plane(Rect(j,i, windowSize,windowSize));
                Mat glcmMat;
                for(int g = 0; g < glcms.size(); g++){
                    const GLCM& glcm = glcms[g];
                    glcm.getGLCMMat(roi, glcmMat);
                    vector<float> feats = computeFeatures(glcmMat);
                    assert(feats.size() == dimensions/(3*glcms.size()));
                    for(int d = 0; d < feats.size(); d++){
                        int columnIndex = j*dimensions + c*glcms.size()*feats.size() + g*feats.size() + d;
                        assert(columnIndex >= 0 && columnIndex < outFeatureImage.cols);
                        outFeatureImage.at<float>(i, columnIndex) = feats[d];
                    }
                }
            }
        }
    }
//    if(DEBUG)
        //debugSaveImage("convolved/glcm.png", outFeatureImage.reshape(dimensions, 0));
    PRINT_DEBUG("GLCM GET FEATURES END!");
}

vector<float> GLCMFeature::computeFeatures(const Mat& p)
{
    vector<float> feats;
    feats.reserve(dimensions);
    float meanI, meanJ, varI, varJ;
    if(useFeat(Contrast))
        feats.push_back(computeContrast(p));
    if(useFeat(Dissimilarity))
        feats.push_back(computeDissimilarity(p));
    if(useFeat(Homogeneity))
        feats.push_back(computeHomogeneity(p));
    if(useFeat(Energy))
        feats.push_back(computeEnergy(p));
    if(useFeat(Entropy))
        feats.push_back(computeEntropy(p));
    if(useFeat(Mean))
        feats.push_back(computeMean(p, &meanI, &meanJ));
    if(useFeat(StdDev)){
        if(!useFeat(Mean))
            computeMean(p, &meanI, &meanJ);
        feats.push_back(computeStdDev(p, meanI, meanJ, &varI, &varJ));
    }
    if(useFeat(Correlation)){
        if(!useFeat(StdDev)){
            if(!useFeat(Mean))
                computeMean(p, &meanI, &meanJ);
            computeStdDev(p, meanI, meanJ, &varI, &varJ);
        }
        feats.push_back(computeCorrelation(p, meanI, meanJ, varI, varJ));
    }
    return feats;
}

float GLCMFeature::computeContrast(const Mat& p)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            sum += pPointer[j]*(i-j)*(i-j);
        }
    }
    return sum;
}

float GLCMFeature::computeDissimilarity(const Mat& p)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            sum += pPointer[j]*std::abs(i-j);
        }
    }
    return sum;
}

float GLCMFeature::computeHomogeneity(const Mat& p)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            sum += pPointer[j]/(1.0+(i-j)*(i-j));
        }
    }
    return sum;
}

float GLCMFeature::computeEnergy(const Mat& p)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            sum += pPointer[j]*pPointer[j];
        }
    }
    return sqrt(sum);
}

float GLCMFeature::computeEntropy(const Mat& p)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            if(pPointer[j] != 0)
                sum += -(pPointer[j])*std::log(pPointer[j]);
        }
    }
    return sum;
}

float GLCMFeature::computeMean(const Mat& p, float* outMeanI, float* outMeanJ)
{
    float sumI = 0;
    float sumJ = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            sumI += pPointer[j]*i;
            sumJ += pPointer[j]*j;
        }
    }
    if(outMeanI)
        *outMeanI = sumI;
    if(outMeanJ)
        *outMeanJ = sumJ;
    return (sumI+sumJ)/2.0;
}

float GLCMFeature::computeStdDev(const Mat& p, float meanI, float meanJ, float* outVarI, float* outVarJ)
{
    float varI = 0;
    float varJ = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            varI += pPointer[j]*(i-meanI)*(i-meanI);
            varJ += pPointer[j]*(j-meanJ)*(j-meanJ);
        }
    }
    if(outVarI)
        *outVarI = varI;
    if(outVarJ)
        *outVarJ = varJ;
    return (sqrt(varI)+sqrt(varJ))/2.0;
}

float GLCMFeature::computeCorrelation(const Mat& p, float meanI, float meanJ, float varI, float varJ)
{
    float sum = 0;
    for(int i = 0; i < p.rows; i++){
        const float* pPointer = p.ptr<float>(i);
        for(int j = 0; j < p.cols; j++){
            float temp = ((i-meanI)*(j-meanJ))/sqrt(varI*varJ);
            sum += pPointer[j]*temp;
        }
    }
    return sum;
}

void GLCMFeature::setUseFeat(uint feat)
{
    featuresToUse |= feat;
    dimensions = std::bitset<sizeof(uint)*8>(featuresToUse).count();
}

void GLCMFeature::setDontUseFeat(uint feat)
{
    featuresToUse &= ~feat;
    dimensions = std::bitset<sizeof(uint)*8>(featuresToUse).count();
}

bool GLCMFeature::useFeat(uint feat)
{
    return featuresToUse&feat;
}




