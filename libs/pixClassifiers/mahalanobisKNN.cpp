#include "mahalanobisKNN.hpp"
#include <omp.h>

MahalanobisKNN::MahalanobisKNN(uint order)
    : order(order)
{
}

MahalanobisKNN::~MahalanobisKNN()
{
    delete mahala;
}

void MahalanobisKNN::train(Mat& samples, Mat& labels)
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
//        for(int c = 0; c < trainData.cols; c++){
//            d[c] = trainData.at<float>(r, c);
//        }
        data[r].v = d;
    }
    pa->setData(data);
    mahala->setPattern(pa);
    PRINT_DEBUG("Making Space.");
    mahala->makeSpace(order);
    PRINT_DEBUG("Making Space Done!");
}

void MahalanobisKNN::classify(Mat& data, Mat& outLabels)
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
        threadDataOutput[i] = new double[trainData.rows*order];
    #pragma omp parallel for num_threads(numThread)
    for(int i = 0; i < dataDouble.rows; i++){
        int threadID = omp_get_thread_num();

        double* dataI = dataDouble.ptr<double>(i);
        double* trainDataPtr = trainData.ptr<double>(0);
        double* dataOutput = threadDataOutput[threadID];
        mahala->evaluateToVector(trainDataPtr, dataI, trainData.rows, trainData.cols, dataOutput);

        double minDist = std::numeric_limits<double>::max();
        int minIndex = -1;
        for(int j = 0; j < trainData.rows; j++){
            int dataOutputIndex = (order-1)*trainData.rows + j;
            double dist = dataOutput[dataOutputIndex];
            if(dist <= minDist){
                minDist = dist;
                minIndex = j;
            }
        }
        outLabels.at<int>(i, 0) = trainLabels.at<int>(minIndex, 0);

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


