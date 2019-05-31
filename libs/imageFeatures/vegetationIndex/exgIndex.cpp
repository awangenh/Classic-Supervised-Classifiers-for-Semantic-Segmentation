#include "exgIndex.hpp"

using namespace std;
using namespace cv;

ExGIndex::ExGIndex()
{
    dimensions = 1;
}

void ExGIndex::getIndex(const Mat &image, vector<Point2i> &pixCoordinates, Mat &outIndexes)
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

    float inMax = 2.0;
    float inMin = -1.0;
    float inRange = inMax-inMin;
    float outMax = 1;
    float outMin = -1;
    float outrange = outMax-outMin;
    for(int i = 0; i < pixCoordinates.size(); i++){
        Point2i coord = pixCoordinates[i];
        Vec3b bgr = image.at<Vec3b>(coord);
        float& index = outIndexes.at<float>(i, 0);

        double b = bgr[0]/255.0;
        double g = bgr[1]/255.0;
        double r = bgr[2]/255.0;

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

        double exg = (2*g-b-r);
        index = (outrange/inRange)*exg + (-outrange*inMin/inRange + outMin);
//        index = exg;
        assert(index >= -1 && index <= 1);
    }
}

void ExGIndex::getIndex(const Mat &image, Mat &outIndexImage)
{
    assert(image.data);
    assert(image.type() == CV_8UC3);

    if(outIndexImage.data == NULL){
        outIndexImage = Mat(image.rows, image.cols*1, CV_32FC1);
    }
    else{
        assert(outIndexImage.rows == image.rows);
        assert(outIndexImage.cols == image.cols*1);
        assert(outIndexImage.type() == CV_32FC1);
    }


    float inMax = 2.0;
    float inMin = -1.0;
    float inRange = inMax-inMin;
    float outMax = 1;
    float outMin = -1;
    float outrange = outMax-outMin;
    for(int i = 0; i < image.rows; i++){
        const Vec3b* pixel = image.ptr<Vec3b>(i);
        float* index = outIndexImage.ptr<float>(i);
        for(int j = 0; j < image.cols; j++){
            double b = pixel[j][0]/255.0;
            double g = pixel[j][1]/255.0;
            double r = pixel[j][2]/255.0;

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

            double exg = (2*g-b-r);
//            index[j] = exg;
            index[j] = (outrange/inRange)*exg + (-outrange*inMin/inRange + outMin);
            assert(index[j] >= -1 && index[j] <= 1);
        }
    }
}
