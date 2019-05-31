#include "ib2Mahala.hpp"
#include <omp.h>
#include <numeric>

using namespace std;
using namespace cv;

IB2Mahala::IB2Mahala(uint order)
    : order(order)
{
}

IB2Mahala::~IB2Mahala()
{
    delete mahala;
}

void IB2Mahala::train(Mat& samples, Mat& labels)
{
    assert(samples.type() == CV_32FC1);
    assert(labels.type() == CV_32SC1);
    assert(labels.cols == 1);
    assert(samples.rows == labels.rows);

    samples.convertTo(trainData, CV_64F);
    trainLabels = labels.clone();

    mahala = new PolyMahala();
    pattern* pa = new pattern(trainData.rows, trainData.cols);
    doubleVector* data = new doubleVector[trainData.rows];
    for(int r = 0; r < trainData.rows; r++){
        double* d = new double[trainData.cols];
        Mat row = trainData.row(r);
        memcpy((void*)d, (void*)row.ptr<double>(0), trainData.cols*sizeof(double));
        data[r].v = d;
    }
    pa->setData(data);
    mahala->setPattern(pa);
    PRINT_DEBUG("Making Space.");
    mahala->makeSpace(order);
    PRINT_DEBUG("Making Space Done!");

    //shurffling the trainData matrix, and the trainLabels in the same order.
    vector<int> randVec(trainData.rows);
    std::iota(randVec.begin(), randVec.end(), 0);//fill with 0,1,2,3,4...
    cv::randShuffle(randVec);
    Mat trainDataShurffled = Mat(trainData.size(), CV_64FC1);
    Mat trainLabelsShurffled = Mat(trainLabels.size(), CV_32SC1);
    for(int i = 0; i < randVec.size(); i++){
        trainData.row(randVec[i]).copyTo(trainDataShurffled.row(i));
        trainLabels.row(randVec[i]).copyTo(trainLabelsShurffled.row(i));
    }
    trainData = trainDataShurffled;
    trainLabels = trainLabelsShurffled;
    //shuffling done!

    //IB2 start here!
    PRINT_DEBUG("IB2 Starting!");
    cd.push_back(trainData.row(0));
    cdLabels.push_back(trainLabels.row(0));
    for(int x = 1; x < trainData.rows; x++){
        //for each X on trainData.
        double* xData = trainData.row(x).ptr<double>(0);
        //find Y, where Y is the closest to X on CD.
        assert(cd.isContinuous());
        double* cdDataPtr = cd.ptr<double>(0);
        //dataOutput contains the distance from all cd elements to X.
        double* dataOutput = new double[cd.rows*order];
        mahala->evaluateToVector(cdDataPtr, xData, cd.rows, cd.cols, dataOutput);
        //this line gets the position of the closest element.
        int orderShift = (order-1)*cd.rows;
        int y = std::distance(dataOutput+orderShift, std::min_element(dataOutput+orderShift, dataOutput+orderShift+cd.rows));
        delete[] dataOutput;
        if(trainLabels.at<int>(x,0) != cdLabels.at<int>(y,0)){
            //classification incorrect, we should memorize this element.
            cd.push_back(trainData.row(x));
            cdLabels.push_back(trainLabels.row(x));
        }
    }
    PRINT_DEBUG("Train data has %d elements.\nCD has %d elements.", trainData.rows, cd.rows);
    PRINT_DEBUG("IB2 Done!");
    //IB2 ends here!

    //debug - visualize the concept descriptor.
    //only work for 2D data, with values ranging from 0 to 1.
    if(DEBUG && cd.cols == 2){
        assert(cd.cols == 2);
        vector<Vec3b> colors;
        colors.push_back(Vec3b(0,0,255));
        colors.push_back(Vec3b(0,255,0));
        colors.push_back(Vec3b(255,0,0));
        colors.push_back(Vec3b(0,255,255));
        colors.push_back(Vec3b(255,0,255));
        colors.push_back(Vec3b(255,255,0));
        Mat debugShowCD = Mat::zeros(1000,1000, CV_8UC3);
        for(int i = 0; i < cd.rows; i++){
            Vec2d& data = cd.at<Vec2d>(i,0);
            assert(data[0] >= 0 && data[0] < 1);
            assert(data[1] >= 0 && data[1] < 1);
            Point p = Point2d(data[0], data[1])*1000;
            cv::circle(debugShowCD, p, 3, Scalar(colors[cdLabels.at<int>(i,0)]), CV_FILLED);
        }
        imshow("Concept Descriptor DEBUG", debugShowCD);
        while(((char)waitKey(1)) != 'q');
    }
    //debug end.

}

void IB2Mahala::classify(Mat& data, Mat& outLabels)
{
    assert(data.data);
    assert(data.type() == CV_32FC1);
    assert(data.cols == trainData.cols);

    static const int numThread = omp_get_max_threads();

    outLabels = Mat(data.rows, 1, CV_32SC1);
    Mat dataDouble;
    data.convertTo(dataDouble, CV_64F);

    vector<int> threadProgress(numThread, 0);
    int mark = dataDouble.rows/10;
    vector<double*> threadDataOutput(numThread, NULL);
    for(int i = 0; i < numThread; i++)
        threadDataOutput[i] = (double*)calloc(cd.rows*(order+1),sizeof(double));

    #pragma omp parallel for num_threads(numThread)
    for(int i = 0; i < dataDouble.rows; i++){
        int threadID = omp_get_thread_num();

        assert(cd.isContinuous());
        double* dataI = dataDouble.ptr<double>(i);
        double* cdDataPtr = cd.ptr<double>(0);
        double* dataOutput = threadDataOutput[threadID];
        mahala->evaluateToVector(cdDataPtr, dataI, cd.rows, cd.cols, dataOutput);

        int orderShift = (order-1)*cd.rows;
        int minIndex = std::distance(dataOutput+orderShift, std::min_element(dataOutput+orderShift, dataOutput+orderShift+cd.rows));
        outLabels.at<int>(i, 0) = cdLabels.at<int>(minIndex, 0);

        threadProgress[threadID]++;
        if(threadID == 0){
            int sum = 0;
            for(int t = 0; t < numThread; t++)
                sum += threadProgress[t];
            if(sum > mark){
                mark += dataDouble.rows/10;
                cout << "Progress => " << sum << " of " << dataDouble.rows << endl;
            }
        }
    }
    for(int i = 0; i < numThread; i++)
        delete[] threadDataOutput[i];
}


