#include "exrIndex.hpp"

ExRIndex::ExRIndex()
{
    dimensions = 1;
}

void ExRIndex::getIndex(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outIndexes)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outIndexes.data == NULL){
        outIndexes = Mat(pixCoordinates.size(), 1, CV_32FC1);
    }
    else{
        assert(outIndexes.rows == pixCoordinates.size());
        assert(outIndexes.cols == 1);
        assert(outIndexes.type() == CV_32FC1);
    }

    float inMax = 1.4;
    float inMin = -1.0;
    float inRange = inMax-inMin;
    float outMax = 1;
    float outMin = -1;
    float outrange = outMax-outMin;
    for(int i = 0; i < pixCoordinates.size(); i++){
        Point2i coord = pixCoordinates[i];
        Vec3b bgr = image.at<Vec3b>(coord);
        float& index = outIndexes.at<float>(i, 0);
        float b = bgr[0]/255.0;
        float g = bgr[1]/255.0;
        float r = bgr[2]/255.0;
        float sum = b+g+r;
        if(sum != 0){
            b = b/(sum);
            g = g/(sum);
            r = r/(sum);
        }
        else{
            b = 0;
            g = 0;
            r = 0;
        }
        float exr = 1.4*r-g;
        index = (outrange/inRange)*exr + (-outrange*inMin/inRange + outMin);
        assert(index >= -1 && index <= 1);
    }
}

void ExRIndex::getIndex(const Mat &image, Mat &outIndexImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outIndexImage.data == NULL){
        outIndexImage = Mat(image.rows, image.cols*1, CV_32FC1);
    }
    else{
        assert(outIndexImage.rows == image.rows);
        assert(outIndexImage.cols == image.cols*dimensions);
        assert(outIndexImage.type() == CV_32FC1);
    }

    float inMax = 1.4;
    float inMin = -1.0;
    float inRange = inMax-inMin;
    float outMax = 1;
    float outMin = -1;
    float outrange = outMax-outMin;
    const Vec3b* pixel = image.ptr<Vec3b>(0);
    float* index = outIndexImage.ptr<float>(0);
    for(int i = 0; i < image.total(); i++){
        float b = pixel[i][0]/255.0;
        float g = pixel[i][1]/255.0;
        float r = pixel[i][2]/255.0;
        float sum = b+g+r;
        if(sum != 0){
            b = b/(sum);
            g = g/(sum);
            r = r/(sum);
        }
        else{
            b = 0;
            g = 0;
            r = 0;
        }
        float exr = 1.4*r-g;
//        index[i] = exr;
        index[i] = (outrange/inRange)*exr + (-outrange*inMin/inRange + outMin);
        assert(index[i] >= -1 && index[i] <= 1);
    }
}
