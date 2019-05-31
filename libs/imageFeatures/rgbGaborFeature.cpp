#include "rgbGaborFeature.hpp"
#include "debug.h"
#include <omp.h>
#include "utils/util.hpp"

using namespace std;
using namespace cv;


RGBGaborFeature::RGBGaborFeature(const Mat &inImage, const GaborTexture &gabor, bool preCalculated)
    : gabor(gabor),
      preCalculated(preCalculated)
{
    assert(inImage.data);
    assert(inImage.depth() == CV_8U);

    dimensions = (1 + this->gabor.kernelsSize())*inImage.channels();

    borderSize = gabor.getKernelMaxSize()/2;

    imageOriginalSize = inImage.size();
    cv::copyMakeBorder(inImage, image, borderSize, borderSize, borderSize, borderSize, BORDER_REFLECT);
    image.convertTo(image, CV_MAKETYPE(CV_32F, image.channels()), (2.0/255.0), -1.0);

    if(DEBUG){
        double max, min;
        cv::minMaxLoc(image, &min, &max);
        assert(max <= 1 && min >= -1);
    }

    if(preCalculated){

        static const int numThread = omp_get_max_threads();

        data = vector<Mat>(dimensions/image.channels());
        data[0] = image;

        if(DEBUG)
            debugSaveImage("gabor/gaborDimensionRGB/gabor"+std::to_string(0)+".png", image);

        static int debugK = 0;

        #pragma omp parallel for num_threads(numThread)
        for(int k = 0; k < gabor.kernelsSize(); k++){
            Mat convolved = Mat(image.size(), CV_MAKETYPE(CV_32F, image.channels()));
            gabor.convolution(image, k, convolved);
            data[k+1] = (convolved);
            if(DEBUG){
                debugSaveImage("gabor/gaborDimensionRGB/gabor"+std::to_string(debugK)+"-"+std::to_string(k+1)+".png", convolved);
                #pragma omp critical (coutLock)
                cout << "index = " << k << " convolution done!" << endl;
            }
        }
        debugK++;
    }
}

RGBGaborFeature::~RGBGaborFeature()
{

}

void RGBGaborFeature::getFeature(std::vector<cv::Point2i>& pixCoordinates, std::vector<double *> &outFeatures)
{

    outFeatures.clear();
    outFeatures = std::vector<double*>(pixCoordinates.size());

    int nChannels = image.channels();

    static const int numThread = omp_get_max_threads();
    #pragma omp parallel for num_threads(numThread)
    for(int i = 0; i < pixCoordinates.size(); i++){
        Point2i p = pixCoordinates[i];
        assert(p.y >=0 && p.y < imageOriginalSize.height);
        assert(p.x >=0 && p.x < imageOriginalSize.width);
        p.y = p.y + borderSize;
        p.x = p.x + borderSize;

        double* piFeatures = new double[dimensions];
        int dimensionIterator = 0;
        if(preCalculated){

            for(Mat& d : data){
                for(int c = 0; c < nChannels; c++){
                    piFeatures[dimensionIterator++] = d.at<float>(p.y, p.x*nChannels+c);
                }
            }
        }
        else{
            for(int c = 0; c < nChannels; c++){
                piFeatures[dimensionIterator++] = (image.at<float>(p.y, p.x*nChannels+c));
            }
            for(int k = 0; k < gabor.kernelsSize(); k++){
                Mat convolved = gabor.convolution(image, k, p);
                for(int c = 0; c < nChannels; c++){
                    piFeatures[dimensionIterator++] = convolved.at<float>(0, c);
                }
            }
        }
        outFeatures[i] = piFeatures;
        assert(dimensionIterator == dimensions);
    }
}

void RGBGaborFeature::getFeature(std::vector<double*>& outFeatures)
{
    outFeatures.clear();
    outFeatures = std::vector<double*>(imageOriginalSize.area());

    int nChannels = image.channels();

    static const int numThread = omp_get_max_threads();
    #pragma omp parallel for collapse(2) num_threads(numThread)
    for(int i = 0; i < imageOriginalSize.height; i++){
        for(int j = 0; j < imageOriginalSize.width; j++){
            Point2i p(j, i);
            p.y = p.y + borderSize;
            p.x = p.x + borderSize;

            assert(p.y >= borderSize && p.y < (image.rows-borderSize));
            assert(p.x >= borderSize && p.x < (image.cols-borderSize));

            double* piFeatures = new double[dimensions];

            int dimensionIterator = 0;
            if(preCalculated){
                for(Mat& d : data){
                    for(int c = 0; c < nChannels; c++){
                        piFeatures[dimensionIterator++] = d.at<float>(p.y, p.x*nChannels+c);
                    }
                }
            }
            else{
                for(int c = 0; c < nChannels; c++){
                    piFeatures[dimensionIterator++] = (image.at<float>(p.y, p.x*nChannels+c));
                }
                for(int k = 0; k < gabor.kernelsSize(); k++){
                    Mat convolved = gabor.convolution(image, k, p);
                    for(int c = 0; c < nChannels; c++){
                        piFeatures[dimensionIterator++] = convolved.at<float>(0, c);
                    }
                }
            }
            outFeatures[(i*imageOriginalSize.width) + j] = piFeatures;
            assert(dimensionIterator == dimensions);
        }
    }
}

void RGBGaborFeature::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
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

    Mat gaborImage;
    gaborFeature->getFeatures(image, pixCoordinates, gaborImage);

    Mat rgbFeatureImage = Mat(pixCoordinates.size(), 3, CV_32FC1);
    for(int i = 0; i < pixCoordinates.size(); i++){
        const Point2i& coord = pixCoordinates[i];
        Vec3b bgr = image.at<Vec3b>(coord);
        Vec3f& feature = rgbFeatureImage.at<Vec3f>(i, 0);
        feature[0] = bgr[0] * (2.0/255.0) + (-1.0);
        feature[1] = bgr[1] * (2.0/255.0) + (-1.0);
        feature[2] = bgr[2] * (2.0/255.0) + (-1.0);
    }

    cv::hconcat(rgbFeatureImage, gaborImage, outFeatures);
    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
}

void RGBGaborFeature::getFeatures(const Mat &image, Mat &outFeatureImage)
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

    Mat gaborImage;
    gaborFeature->getFeatures(image, gaborImage);
    gaborImage = gaborImage.reshape(1, image.total());
    assert(gaborImage.cols == gaborFeature->getDimentions());
    assert(gaborImage. rows == image.total());

    Mat imageTemp = image.reshape(1, image.total());
    Mat rgbFeatureImage;
    imageTemp.convertTo(rgbFeatureImage, CV_32F, (2.0/255.0), -1.0);
    assert(rgbFeatureImage.cols == 3);
    assert(rgbFeatureImage.rows == image.total());

    cv::hconcat(rgbFeatureImage, gaborImage, outFeatureImage);
    outFeatureImage = outFeatureImage.reshape(1, image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
    assert(outFeatureImage.rows = image.rows);
}


