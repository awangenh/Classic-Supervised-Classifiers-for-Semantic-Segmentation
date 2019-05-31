#include "rgbVegetationIndex.hpp"

using namespace std;
using namespace cv;

RGBVegetationIndex::RGBVegetationIndex(VegetationIndex *vIndex)
    : ImageFeatures(), vegetationIndex(vIndex)
{
    dimensions = 3 + vegetationIndex->getDimentions();
}

RGBVegetationIndex::~RGBVegetationIndex()
{

}

void RGBVegetationIndex::getFeature(vector<Point2i> &pixCoordinates, vector<double *> &outFeatures)
{

}

void RGBVegetationIndex::getFeature(vector<double *> &outFeatures)
{

}

void RGBVegetationIndex::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
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

    Mat indexImage;
    vegetationIndex->getIndex(image, pixCoordinates, indexImage);

    Mat rgbFeatureImage = Mat(pixCoordinates.size(), 3, CV_32FC1);
    for(int i = 0; i < pixCoordinates.size(); i++){
        const Point2i& coord = pixCoordinates[i];
        Vec3b bgr = image.at<Vec3b>(coord);
        Vec3f& feature = rgbFeatureImage.at<Vec3f>(i, 0);
        feature[0] = bgr[0] * (2.0/255.0) + (-1.0);
        feature[1] = bgr[1] * (2.0/255.0) + (-1.0);
        feature[2] = bgr[2] * (2.0/255.0) + (-1.0);
    }

    cv::hconcat(rgbFeatureImage, indexImage, outFeatures);
    assert(outFeatures.cols == dimensions);
    assert(outFeatures.rows == pixCoordinates.size());
}

void RGBVegetationIndex::getFeatures(const Mat& image, Mat& outFeatureImage)
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

    Mat indexImage;
    vegetationIndex->getIndex(image, indexImage);
    indexImage = indexImage.reshape(1, image.total());
    assert(indexImage.cols == vegetationIndex->getDimentions());
    assert(indexImage. rows == image.total());

    Mat imageTemp = image.reshape(1, image.total());
    Mat rgbFeatureImage;
    imageTemp.convertTo(rgbFeatureImage, CV_32F, (2.0/255.0), -1.0);
    assert(rgbFeatureImage.cols == 3);
    assert(rgbFeatureImage.rows == image.total());

    cv::hconcat(rgbFeatureImage, indexImage, outFeatureImage);
    outFeatureImage = outFeatureImage.reshape(1, image.rows);
    assert(outFeatureImage.cols = image.cols*dimensions);
    assert(outFeatureImage.rows = image.rows);
}
