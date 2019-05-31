#include "vegetationIndex.hpp"

using namespace std;
using namespace cv;

void VegetationIndex::getFeature(vector<Point2i> &pixCoordinates, vector<double *> &outFeatures)
{

}

void VegetationIndex::getFeature(vector<double *> &outFeatures)
{

}

void VegetationIndex::getFeatures(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outFeatures)
{
    getIndex(image, pixCoordinates, outFeatures);
}

void VegetationIndex::getFeatures(const Mat &image, Mat &outFeatureImage)
{
    getIndex(image, outFeatureImage);
}
