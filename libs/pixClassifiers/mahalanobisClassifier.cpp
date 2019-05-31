#include "mahalanobisClassifier.hpp"
#include <omp.h>

MahalanobisClassifier::MahalanobisClassifier(uint order)
    : order(order)
{
}

MahalanobisClassifier::~MahalanobisClassifier()
{
    for(PolyMahala* p : mahalas){
        delete p;
    }
}

void MahalanobisClassifier::train(Mat& samples, Mat& labels)
{
    assert(samples.type() == CV_32FC1);
    assert(labels.type() == CV_32SC1);
    assert(labels.cols == 1);
    assert(samples.rows == labels.rows);

    std::map<int, int> labelsMap;
    for(int i = 0; i < labels.rows; i++){
        int classLabel = labels.at<int>(i, 0);
        labelsMap[classLabel]++;
    }
    assert(labelsMap.size() > 0);

    vector<Mat> classesSamples;
    vector<Mat> classesLabels;
    for(const auto& pair : labelsMap){
        int label = pair.first;
        int numberOfSamples = pair.second;
        Mat samplesI = Mat(numberOfSamples, samples.cols, CV_32FC1);
        Mat labelsI = Mat(numberOfSamples, 1, CV_32SC1, Scalar(label));
        int count = 0;
        for(int i = 0; i < samples.rows; i++){
            int l = labels.at<int>(i, 0);
            if(label == l){
                samples.row(i).copyTo(samplesI.row(count++));
            }
        }
        assert(count == samplesI.rows);
        labelsNumbers.push_back(label);
        classesSamples.push_back(samplesI);
        classesLabels.push_back(labelsI);
    }

    int debugCount = 0;
    for(int i = 0; i < classesSamples.size(); i++){
        mahalas.push_back(new PolyMahala());
        Mat s = classesSamples[i];
        Mat l = classesLabels[i];
        debugCount += s.rows;
        pattern* pa = new pattern(s.rows, s.cols);
        doubleVector* data = new doubleVector[s.rows];
        for(int r = 0; r < s.rows; r++){
            double* d = new double[s.cols];
            for(int c = 0; c < s.cols; c++){
                d[c] = s.at<float>(r, c);
            }
            data[r].v = d;
        }
        pa->setData(data);
        mahalas[i]->setPattern(pa);
        mahalas[i]->makeSpace(order);
    }
    assert(debugCount == samples.rows);
    assert(mahalas.size() == labelsNumbers.size());
}

void MahalanobisClassifier::classify(Mat& data, Mat& outLabels)
{
    assert(data.data);
    assert(data.type() == CV_32FC1);

    static const int numThread = omp_get_max_threads();

    outLabels = Mat(data.rows, 1, CV_32SC1);

    vector<int> threadProgress(numThread, 0);
    int mark = data.rows/10;
    vector<double*> threadDataOutput(numThread, NULL);
    vector<double*> threadDataInput(numThread, NULL);
    for(int i = 0; i < numThread; i++){
        threadDataOutput[i] = new double[1*order];
        threadDataInput[i] = new double[data.cols];
    }
    #pragma omp parallel for num_threads(numThread)
    for(int i = 0; i < data.rows; i++){
        int threadID = omp_get_thread_num();
        float* inFloat = data.ptr<float>(i);
        double* in = threadDataInput[threadID];
        for(int j = 0; j < data.cols; j++)
            in[j] = inFloat[j];
        double minDist = std::numeric_limits<double>::max();
        int outLabel = -1;
        for(int l = 0; l < mahalas.size(); l++){
            double* dataOutput = threadDataOutput[threadID];
            mahalas[l]->evaluateToCenter(in, 1, data.cols, dataOutput);
            double dist = dataOutput[order-1];
            if(dist <= minDist){
                minDist = dist;
                outLabel = l;
            }
        }
        outLabels.at<int>(i, 0) = labelsNumbers[outLabel];

        threadProgress[threadID]++;
        if(threadID == 0){
            int sum = 0;
            for(int t = 0; t < numThread; t++)
                sum += threadProgress[t];
            if(sum > mark){
                mark += data.rows/10;
                cout << "Progress => " << sum << " of " << data.rows << endl;
            }
        }
    }
    for(int i = 0; i < numThread; i++){
        delete[] threadDataOutput[i];
        delete[] threadDataInput[i];
    }
}


