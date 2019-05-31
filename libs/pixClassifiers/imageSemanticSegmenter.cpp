#include "imageSemanticSegmenter.hpp"

#include <omp.h>
#include <ctime>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <deque>
#include <math.h>
#include <iomanip>
#include <fstream>

using namespace std;
using namespace cv;

ImageSemanticSegmenter::ImageSemanticSegmenter(PixelClassifier* classifier)
    : classifier(classifier)
{

}

ImageSemanticSegmenter::~ImageSemanticSegmenter()
{
    delete classifier;
}

void ImageSemanticSegmenter::doSegmentation(Mat& inImage, Mat& outMap)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    classifier->run(inImage, outMap);
    assert(outMap.type() == CV_32SC1);
}
