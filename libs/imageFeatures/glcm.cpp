#include "glcm.hpp"
#include "debug.h"

using namespace std;
using namespace cv;

GLCM::GLCM(uint dirs, uint depth)
{
    assert(depth >= 2 && depth <= 8);
    this->depth = depth;
    if(dirs & NORTH)
        directions.push_back(Point(0,-1));
    if(dirs & SOUTH)
        directions.push_back(Point(0,1));
    if(dirs & EAST)
        directions.push_back(Point(1,0));
    if(dirs & WEST)
        directions.push_back(Point(-1,0));
    if(dirs & NORTH_EAST)
        directions.push_back(Point(1,-1));
    if(dirs & NORTH_WEST)
        directions.push_back(Point(-1,-1));
    if(dirs & SOUTH_EAST)
        directions.push_back(Point(1,1));
    if(dirs & SOUTH_WEST)
        directions.push_back(Point(-1,1));
}

GLCM::GLCM(const GLCM &other)
    : directions(other.directions), depth(other.depth)
{

}

GLCM::~GLCM()
{

}

void GLCM::getGLCMMat(const cv::Mat &inImage, Mat &outMat) const
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC1);

    int levels = pow(2, depth);
    if(outMat.data == NULL){
        outMat = Mat::zeros(levels, levels, CV_32FC1);
    }
    else{
        assert(outMat.rows == levels);
        assert(outMat.cols == levels);
        assert(outMat.type() == CV_32FC1);
        outMat.setTo(0);
    }

    uchar depthDivisor = pow(2, 8-depth);
    uint count = 0;
    for(int i = 0; i < inImage.rows; i++){
        const uchar* inPointer = inImage.ptr<uchar>(i);
        for(int j = 0; j < inImage.cols; j++){
            Point pRef = Point(j,i);
            uchar valRef = inPointer[j]/depthDivisor;
            for(int d = 0; d < directions.size(); d++){
                Point pNeighbour = pRef+directions[d];
                bool isInside = (pNeighbour.x >= 0 && pNeighbour.x < inImage.cols &&
                                 pNeighbour.y >= 0 && pNeighbour.y < inImage.rows);
                if(isInside){
                    uchar valNeighbour = inImage.at<uchar>(pNeighbour)/depthDivisor;
                    assert(valRef < levels);
                    assert(valNeighbour < levels);
                    outMat.at<float>(valRef,valNeighbour) += 1;
                    count++;
                }
            }
        }
    }
    outMat /= count;
}
