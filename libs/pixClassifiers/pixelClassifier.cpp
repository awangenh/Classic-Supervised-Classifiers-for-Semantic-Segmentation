#include "pixelClassifier.hpp"
#include "debug.h"

#include <omp.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

PixelClassifier::PixelClassifier(ImageFeatures* calculator)
    : featureCalculator(calculator)
{
}

PixelClassifier::~PixelClassifier()
{

}

bool PixelClassifier::loadTrainData(string path)
{
    FileStorage file(path, FileStorage::READ);
    if(!file.isOpened()){
        PRINT_DEBUG("Canot opencv file %s", path);
        return false;
    }

    int numberOfClasses;
    int numberOfLabeledImages;
    int numberOfAnotatedImages;
    file["numberOfClasses"] >> numberOfClasses;
    file["numberOfLabeled"] >> numberOfLabeledImages;
    file["numberOfAnotated"] >> numberOfAnotatedImages;
    PRINT_DEBUG("\n\tNumber of classes = %d \n\tNumber of Labeled Image = %d \n\tNumber of Anotated Image = %d",
                numberOfClasses, numberOfLabeledImages, numberOfAnotatedImages);

    for(int i = 0; i < numberOfLabeledImages; i++){
        string imagePath;
        string labelPath;
        file["labelSample"+to_string(i)] >> imagePath;
        file["label"+to_string(i)] >> labelPath;
        Mat imageI = imread(imagePath);
        Mat labelI = imread(labelPath, CV_LOAD_IMAGE_GRAYSCALE);
        assert(imageI.data);
        assert(labelI.data);
        assert(imageI.size() == labelI.size());
        assert(imageI.type() == CV_8UC3);
        assert(labelI.type() == CV_8UC1);
        labelI.convertTo(labelI, CV_32SC1);
        addTrainData(imageI, labelI);
    }
    for(int i = 0; i < numberOfAnotatedImages; i++){
        string imagePath;
        file["anotationSample"+to_string(i)] >> imagePath;
        Mat imageI = imread(imagePath);
        assert(imageI.data);
        assert(imageI.type() == CV_8UC3);
        vector<vector<Point> > coords;
        for(int n = 0; n < numberOfClasses; n++){
            Mat c;
            vector<Point> vec;
            string name = "anotation"+to_string(i)+"-"+to_string(n);
            file[name] >> c;
            if(c.data){
                assert(c.cols == 2);
                assert(c.type() == CV_32SC1);
                vec.reserve(c.rows);
                for(int r = 0; r < c.rows; r++){
                    Vec2i v = c.at<Vec2i>(r, 0);
                    Point p(v[1], v[0]);
                    vec.push_back(p);
                }
            }
            coords.push_back(vec);
        }
        addTrainData(imageI, coords);
    }
}

void PixelClassifier::run(Mat &inImage, Mat &outLabelImage)
{
    assert(inImage.data);
    assert(inImage.type() == CV_8UC3);

    if(outLabelImage.data){
        assert(outLabelImage.size() == inImage.size());
        assert(outLabelImage.type() == CV_32SC1);
    }else{
        outLabelImage = Mat(inImage.size(), CV_32SC1, Scalar(-1));
    }

    static const long int maxMemory = 8L * 1024L * 1024L * 1024L;//bytes

    long int inImageRows = inImage.rows;
    long int inImageCols = inImage.cols;
    long int numberOfFeatures = featureCalculator->getDimentions();
    long int totalValues = inImageRows * inImageCols * numberOfFeatures;
    long int totalMemory =  totalValues * (long int)sizeof(float);

    if(totalValues > std::numeric_limits<int>::max() || totalMemory > maxMemory){
        //Process window by window.
        static const int windowSize = sqrt(std::numeric_limits<int>::max()/((float)numberOfFeatures));
        Mat outWindow;
        for(int i = 0; i < inImage.rows; i+=windowSize){
            for(int j = 0; j < inImage.cols; j+=windowSize){
                Point rectPoint(j, i);
                Size rectSize(std::min(windowSize, inImage.cols-j), std::min(windowSize, inImage.rows-i));
                Rect windowRect(rectPoint, rectSize);
                Mat inWindow = inImage(windowRect);
                PRINT_DEBUG("CLASSIFYING THE WINDOW %dx%d-%dx%d", windowRect.x, windowRect.y, windowRect.width, windowRect.height);
                classify(inWindow, outWindow);
                outWindow.copyTo(outLabelImage(windowRect));
                PRINT_DEBUG("CLASSIFYING WINDOW DONE!");
            }
        }
    }else{
        classify(inImage, outLabelImage);
    }

}


