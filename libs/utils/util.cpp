#include "util.hpp"
#include <fstream>
#include <iomanip>
#include <experimental/filesystem>

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

void debugSaveImage(string path, const Mat &inImage)
{
    assert(inImage.data);
    assert(inImage.depth() == CV_8U || inImage.depth() == CV_32F || inImage.depth() == CV_64F);
//    assert(inImage.channels() == 1 || inImage.channels() == 3);

    if (inImage.depth() == CV_8U) {
        imwrite(path, inImage);
        return;
    }
    if(inImage.channels() == 1){
        Mat outImage = Mat(inImage.size(), CV_8UC1);
        double max;
        double min;
        cv::minMaxLoc(inImage, &min, &max);
        double range = max-min;
        inImage.convertTo(outImage, CV_8UC1, 255.0/range, -(255.0*min/range));
        imwrite(path, outImage);
        return;
    }
    else{//2 or more channels
        Mat outImage;
        vector<Mat> channels;
        cv::split(inImage, channels);
        int c = 0;
        for(Mat& plane : channels){
            cv::normalize(plane, plane, 0,255, cv::NORM_MINMAX);
            fs::path p = path;
            string filename = p.stem().c_str();
            filename += "channel" + to_string(c++) + (p.extension().c_str());
            p.replace_filename(filename);
            imwrite(p.c_str(), plane);
        }
    }

}

