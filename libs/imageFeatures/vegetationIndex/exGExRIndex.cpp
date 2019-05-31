#include "exGExRIndex.hpp"
#include "exgIndex.hpp"
#include "exrIndex.hpp"

ExGExRIndex::ExGExRIndex(int method)
    : method(method)
{
    if(method == Subtraction){
        dimensions = 1;
    }
    else if(method == biDimesion){
        dimensions = 2;
    }
}

void ExGExRIndex::getIndex(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outIndexes)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outIndexes.data == NULL){
        outIndexes = Mat(pixCoordinates.size(), dimensions, CV_32FC1);
    }
    else{
        assert(outIndexes.rows == pixCoordinates.size());
        assert(outIndexes.cols == dimensions);
        assert(outIndexes.type() == CV_32FC1);
    }

    Mat exgMat, exrMat;
    ExGIndex exg;
    ExRIndex exr;
    exg.getIndex(image, pixCoordinates, exgMat);
    exr.getIndex(image, pixCoordinates, exrMat);

    if(method == Subtraction){
        outIndexes = exgMat-exrMat;
    }else if(method == biDimesion){
        hconcat(exgMat, exrMat, outIndexes);
    }
}

void ExGExRIndex::getIndex(const Mat &image, Mat &outIndexImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outIndexImage.data == NULL){
        outIndexImage = Mat(image.rows, image.cols*dimensions, CV_32FC1);
    }
    else{
        assert(outIndexImage.rows == image.rows);
        assert(outIndexImage.cols == image.cols*dimensions);
        assert(outIndexImage.type() == CV_32FC1);
    }

    Mat exgMat, exrMat;
    ExGIndex exg;
    ExRIndex exr;
    exg.getIndex(image, exgMat);
    exr.getIndex(image, exrMat);

    if(method == Subtraction){
        outIndexImage = exgMat-exrMat;
    }else if(method == biDimesion){
        exgMat = exgMat.reshape(1, image.total());
        exrMat = exrMat.reshape(1, image.total());
        outIndexImage = outIndexImage.reshape(1, image.total());
        hconcat(exgMat, exrMat, outIndexImage);
        outIndexImage = outIndexImage.reshape(1, image.rows);
    }
}
