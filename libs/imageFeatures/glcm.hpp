#pragma once

#include "opencv2/opencv.hpp"

class GLCM{
public:
    enum Dirs{NORTH = 1U, SOUTH = 2U, EAST = 4U, WEST = 8U,
         NORTH_EAST = 16U, NORTH_WEST = 32U,
         SOUTH_EAST = 64U, SOUTH_WEST = 128U,
         ALL = NORTH|SOUTH|EAST|WEST|
               NORTH_EAST|NORTH_WEST|
               SOUTH_EAST|SOUTH_WEST};
    GLCM(uint dirs = ALL, uint depth = 4);
    GLCM(const GLCM& other);
    virtual ~GLCM();

    void getGLCMMat(const cv::Mat& inImage, cv::Mat& outMat) const;

private:
    std::vector<cv::Point> directions;
    uint depth;
};
