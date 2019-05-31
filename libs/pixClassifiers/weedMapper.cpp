#include "weedMapper.hpp"

using namespace std;
using namespace cv;

WeedMapper::WeedMapper(PixelClassifier *classifier)
    : ImageMapper(classifier)
{

}

WeedMapper::~WeedMapper()
{

}

void WeedMapper::setTrainData(Mat &sampleImage, vector<Point2i> &plantSamplePixCoords, vector<Point2i> &weedSamplePixCoords, vector<Point2i> &soilSamplePixCoords)
{
    vector<vector<Point2i> > samplesCords = {plantSamplePixCoords, weedSamplePixCoords, soilSamplePixCoords};
    classifier->setTrainData(sampleImage, samplesCords);
}

void WeedMapper::doWeedMapping(Mat &inImage, Mat &outWeedMap)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    Mat map;
    doMapping(inImage, map);

    Vec3b colorMap[3] = {{  0,   0, 255},
                         {  0, 255, 255},
                         {  0, 255,   0}};

    assert(map.cols == inImage.cols && map.rows == inImage.rows);
    outWeedMap = Mat(inImage.size(), CV_8UC3);
    for(int i = 0; i < inImage.rows; i++){
        for(int j = 0; j < inImage.cols; j++){
            float label = map.at<float>(i, j);
            assert(label == 0 || label == 1 || label == 2);
            outWeedMap.at<Vec3b>(i, j) = colorMap[(int)label];
        }
    }

}
