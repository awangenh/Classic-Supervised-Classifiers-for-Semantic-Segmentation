#include "imgSegPrecisionMeasure.hpp"

using namespace std;
using namespace cv;

ImgSegPrecisionMeasure::ImgSegPrecisionMeasure(int numberOfClass)
    : numberOfClass(numberOfClass)
{
    assert(numberOfClass >= 2);
}

ImgSegPrecisionMeasure::~ImgSegPrecisionMeasure()
{

}

void ImgSegPrecisionMeasure::fillConfusionMat(const Mat &predImg, const Mat &gtImg, Mat &inOutConfusionMat)
{
    assert(predImg.data && gtImg.data);
    assert(predImg.size() == gtImg.size());
    assert(predImg.type() == gtImg.type());
    assert(predImg.type() == CV_32SC1);
    assert(inOutConfusionMat.data);
    assert(inOutConfusionMat.rows == inOutConfusionMat.cols);
    assert(inOutConfusionMat.rows == numberOfClass);
    assert(inOutConfusionMat.type() == CV_32SC1);

    for(int i = 0; i < predImg.rows; i++){
        const int* predData = predImg.ptr<int>(i);
        const int* gtData = gtImg.ptr<int>(i);
        for(int j = 0; j < predImg.cols; j++){
            int predLabel = predData[j];
            int gtLabel = gtData[j];
            assert(predLabel >= -1 && predLabel < numberOfClass);
            assert(gtLabel   >= -1 && gtLabel < numberOfClass);
            if(predLabel > -1 && gtLabel > -1)
                inOutConfusionMat.at<int>(predLabel, gtLabel)++;
        }
    }
}

