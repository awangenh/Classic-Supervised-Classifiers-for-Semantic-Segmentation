#include <iostream>
#include <memory>
#include <set>
#include <experimental/filesystem>

#include <opencv2/opencv.hpp>

#include <boost/program_options.hpp>

#include "pixClassifiers/imageSemanticSegmenter.hpp"
#include "pixClassifiers/pixelKNNClassifier.hpp"
#include "pixClassifiers/pixelSVMClassifier.hpp"
#include "pixClassifiers/pixelMahalanobisClassifier.hpp"
#include "pixClassifiers/pixelNNClassifier.hpp"
#include "pixClassifiers/pixelRandomForestClassifier.hpp"
#include "imageFeatures/multiFeatures.hpp"
#include "imageFeatures/filterFeature.hpp"
#include "imageFeatures/rgbFeature.hpp"
#include "imageFeatures/luminanceFeatures.hpp"
#include "imageFeatures/vegetationIndex/exgIndex.hpp"
#include "imageFeatures/vegetationIndex/exrIndex.hpp"
#include "imageFeatures/vegetationIndex/exGExRIndex.hpp"
#include "imageFeatures/rgbVegetationIndex.hpp"
#include "imageFeatures/gaborFeatures.hpp"
#include "imageFeatures/glcm.hpp"
#include "imageFeatures/glcmFeatures.hpp"

#include "utils/util.hpp"
#include "debug.h"

using namespace std;
using namespace cv;
namespace fs = std::experimental::filesystem::v1;

boost::program_options::options_description desc("Allowed options");
boost::program_options::variables_map vm;

Size interfaceSize;
double interfaceScaleFactor;

string inPaths;
vector<fs::path> inImagesPaths;
string outImagePath = "out/";
uint numberOfClasses;
string classifierConfFilePath;
string featuresConfFilePath;
string trainDataFilePath;
string colorClassesFilePath;

int useMahala = 1;
int useSVM = 0;
int useKNN = 0;
int useNN = 0;
int useRTree = 0;

int mahalaOrder = 3;
int svm_linear = 1;
int svm_rbf = 0;
int knn_k = 3;
int rtree_n_tree = 50;
int rtree_max_deep = 30;
int rtree_max_categories = 16;

int useRGB = 1;
int useEXG = 0;
int useEXR = 0;
int useEXGEXR = 0;
int useGaborFilter = 0;
int useGLCM = 0;
uint glcmFeats = 1;
int glcmWindowSize = 9;
int glcmContrast = 1;
int glcmDissimilarity = 0;
int glcmHomogeneity = 0;
int glcmEnergy = 0;
int glcmEntropy = 0;
int glcmMean = 0;
int glcmStdDev = 0;
int glcmCorrelation = 0;

ImageSemanticSegmenter* mapper;
PixelClassifier* pixClassifier;
ImageFeatures* features;

Mat classesColors;

void mountParams(int argc, char **argv);
void readParams();
void readInImagesPaths();
void readClassifierConfFile();
void readFeatConfFile();
void readColorClassesFile();
void setUpImage();

int main(int argc, char** argv){
    mountParams(argc, argv);
    readParams();

    readInImagesPaths();
    readFeatConfFile();
    readClassifierConfFile();
    readColorClassesFile();

    PRINT_DEBUG("Loading Train Data!");
    pixClassifier->loadTrainData(trainDataFilePath);
    PRINT_DEBUG("Start Training!");
    pixClassifier->train();
    PRINT_DEBUG("Finished Training!");
    mapper = new ImageSemanticSegmenter(pixClassifier);

    system("echo start >> date.txt && date >> date.txt");

    Mat inImage;
    Mat labelImage;
    Mat colorLabelImage;
    for(fs::path& inImagePath : inImagesPaths){

        inImage = imread(inImagePath.c_str());
//        Mat inImage = imread(inImagePath, cv::IMREAD_LOAD_GDAL);
        if(!inImage.data){
            PRINT_DEBUG("ERROR WHILE OPENING IMAGE: %s", inImagePath);
            exit(-1);
        }
//        gdalIO.loadGeoTransform(inImagePath);
//        PRINT_DEBUG("image opened");
//        PRINT_DEBUG("Input image size = %dx%d", inImage.cols, inImage.rows);
//        PRINT_DEBUG("Input image channels and depth = %d - %d", inImage.channels(), inImage.depth());

        mapper->doSegmentation(inImage, labelImage);
        labelImage.convertTo(colorLabelImage, CV_8UC1);
        cvtColor(colorLabelImage, colorLabelImage, COLOR_GRAY2BGR);
        for(int i = 0; i < inImage.rows; i++){
            for(int j = 0; j < inImage.cols; j++){
                int label = labelImage.at<int>(i, j);
                assert(label >= 0 && label < numberOfClasses);
                colorLabelImage.at<Vec3b>(i, j) = classesColors.at<Vec3b>(label, 0);
            }
        }
        string outPath(outImagePath+inImagePath.stem()+"out.png");
        imwrite(outPath, colorLabelImage);
        PRINT_DEBUG("Saved Image = %s", outPath.c_str());
    }

    system("echo end >> date.txt && date >> date.txt");
}

void mountParams(int argc, char** argv){
    namespace po = boost::program_options;
    desc.add_options()
            ("help", "describe arguments")
            ("in", po::value<string>(), "input file name")
            ("out", po::value<string>(), "output file name")
            ("n", po::value<unsigned int>(), "number of classes")
            ("classifier", po::value<string>(), "path to classifier conf file")
            ("features", po::value<string>(), "path to features conf file")
            ("tdata", po::value<string>(), "path to train data conf file")
            ("colors", po::value<string>(), "path to classes colors conf file");

    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);
}

void readParams(){
    if(vm.count("help")){
        cout << desc << endl;
        exit(0);
    }
    if(!vm.count("in")){
        cout << "Missing input file" << endl;
        exit(-1);
    }
    if(!vm.count("n")){
        cout << "Missing number of classes" << endl;
        exit(-1);
    }
    if(!vm.count("colors")){
        cout << "Missing classes colors conf file" << endl;
        exit(-1);
    }
    if(!vm.count("features")){
        cout << "Missing features conf file" << endl;
        exit(-1);
    }
    if(!vm.count("classifier")){
        cout << "Missing classifier conf file" << endl;
        exit(-1);
    }
    inPaths = vm["in"].as<string>();
    outImagePath = vm.count("out") ? vm["out"].as<string>() : "out/";
    numberOfClasses = vm["n"].as<unsigned int>();
    classifierConfFilePath = vm["classifier"].as<string>();
    featuresConfFilePath = vm["features"].as<string>();
    colorClassesFilePath = vm["colors"].as<string>();
    trainDataFilePath = vm.count("tdata") ? vm["tdata"].as<string>() : "";
}

void readInImagesPaths(){
    string extension = inPaths.substr(inPaths.find_last_of(".") + 1);
    assert(extension == "png" || extension == "tif" || extension == "yml");
    if(extension == "yml"){
        cv::FileStorage file(inPaths, cv::FileStorage::READ);
        assert(file.isOpened());
        int numberOfImages;
        file["numberOfImages"] >> numberOfImages;
        for(int i = 0; i < numberOfImages; i++){
            string path;
            file["image"+to_string(i)] >> path;
            inImagesPaths.push_back(fs::path(path));
        }
    }
    else{
        inImagesPaths.push_back(fs::path(inPaths));
    }
}

void readClassifierConfFile(){
    cv::FileStorage file(classifierConfFilePath, cv::FileStorage::READ);
    file["useMAHALANOBIS"] >> useMahala;
    file["useSVM"] >> useSVM;
    file["useKNN"] >> useKNN;
    file["useNN"] >> useNN;
    file["useRTREE"] >> useRTree;

    file["SVM_LINEAR"] >> svm_linear;
    file["SVM_RBF"] >> svm_rbf;

    file["MAHALANOBIS_ORDER"] >> mahalaOrder;

    file["KNN_K"] >> knn_k;

    file["RTREE_N_TREE"] >> rtree_n_tree;
    file["RTREE_MAX_DEEP"] >> rtree_max_deep;
    file["RTREE_MAX_CATEGORIES"] >> rtree_max_categories;

    assert(useMahala || useSVM || useKNN || useNN || useRTree);
    if(useMahala){
        assert(mahalaOrder >= 1);
        pixClassifier = new PixelMahalanobisClassifier(features, mahalaOrder);
    }else if(useSVM){
        assert(svm_linear || svm_rbf);
        if(svm_linear)
            pixClassifier = new PixelSVMClassifier(features, PixelSVMClassifier::LINEAR);
        else if(svm_rbf)
            pixClassifier = new PixelSVMClassifier(features, PixelSVMClassifier::RBF);
    }else if(useKNN){
        assert(knn_k >= 1);
        pixClassifier = new PixelKNNClassifier(features, knn_k);
    }else if(useNN){
        pixClassifier = new PixelNNClassifier(features);
    }else if(useRTree){
        assert(rtree_n_tree >= 1 && rtree_max_deep >= 1 && rtree_max_categories >= 1);
        pixClassifier = new PixelRandomForestClassifier(features, rtree_n_tree, rtree_max_deep, rtree_max_categories);
    }

}

void readFeatConfFile(){
    cv::FileStorage file(featuresConfFilePath, cv::FileStorage::READ);
    file["useRGB"] >> useRGB;
    file["useEXG"] >> useEXG;
    file["useEXR"] >> useEXR;
    file["useEXGEXR"] >> useEXGEXR;
    file["useGaborFilter"] >> useGaborFilter;
    file["useGLCM"] >> useGLCM;

    assert(useRGB || useEXG || useEXR || useEXGEXR || useGLCM);

    MultiFeature* multiF = new MultiFeature();
    if(useRGB){
        multiF->concatFeat(new RGBFeature());
    }
    if(useEXG){
        multiF->concatFeat(new ExGIndex());
    }
    if(useEXR){
        multiF->concatFeat(new ExRIndex);
    }
    if(useEXGEXR){
        multiF->concatFeat(new ExGExRIndex(ExGExRIndex::Subtraction));
    }
    if(useGLCM){
        file["glcmContrast"] >> glcmContrast;
        file["glcmDissimilarity"] >> glcmDissimilarity;
        file["glcmHomogeneity"] >> glcmHomogeneity;
        file["glcmEnergy"] >> glcmEnergy;
        file["glcmEntropy"] >> glcmEntropy;
        file["glcmMean"] >> glcmMean;
        file["glcmStdDev"] >> glcmStdDev;
        file["glcmCorrelation"] >> glcmCorrelation;

        file["glcmWindowSize"] >> glcmWindowSize;

        if(glcmContrast)
            glcmFeats |= GLCMFeature::Contrast;
        if(glcmDissimilarity)
            glcmFeats |= GLCMFeature::Dissimilarity;
        if(glcmHomogeneity)
            glcmFeats |= GLCMFeature::Homogeneity;
        if(glcmEnergy)
            glcmFeats |= GLCMFeature::Energy;
        if(glcmEntropy)
            glcmFeats |= GLCMFeature::Entropy;
        if(glcmMean)
            glcmFeats |= GLCMFeature::Mean;
        if(glcmStdDev)
            glcmFeats |= GLCMFeature::StdDev;
        if(glcmCorrelation)
            glcmFeats |= GLCMFeature::Correlation;
        vector<GLCM> glcms = {GLCM(GLCM::NORTH|GLCM::SOUTH),
                              GLCM(GLCM::EAST|GLCM::WEST)/*,
                              GLCM(GLCM::NORTH_EAST|GLCM::SOUTH_WEST),
                              GLCM(GLCM::NORTH_WEST|GLCM::SOUTH_EAST)*/};
//        vector<GLCM> glcms = {GLCM(GLCM::ALL)};
        multiF->concatFeat(new GLCMFeature(glcms, glcmWindowSize, glcmFeats));
    }

    features = multiF;

    if(useGaborFilter){
        vector<double> thetas;
        thetas.push_back(0 * M_PI/180.0);//here
//        thetas.push_back(15 * M_PI/180.0);
//        thetas.push_back(30 * M_PI/180.0);
//        thetas.push_back(45 * M_PI/180.0);
//        thetas.push_back(60 * M_PI/180.0);
//        thetas.push_back(75 * M_PI/180.0);
        thetas.push_back(90 * M_PI/180.0);//here
//        thetas.push_back(105 * M_PI/180.0);
//        thetas.push_back(120 * M_PI/180.0);
//        thetas.push_back(135 * M_PI/180.0);
//        thetas.push_back(150 * M_PI/180.0);
//        thetas.push_back(165 * M_PI/180.0);
        vector<int> waveLengths;
//        waveLengths.push_back(3);
        waveLengths.push_back(9);//here
        waveLengths.push_back(15);//here
//        waveLengths.push_back(21);
//        waveLengths.push_back(27);
//        waveLengths.push_back(33);
//        waveLengths.push_back(39);
//        waveLengths.push_back(51);
        GaborTexture gabor(thetas, waveLengths);
//        GaborTexture gabor(waveLengths, GaborTexture::CIRCULAR);
        features = new GaborFeature(gabor, features);
//        multiF->concatFeat(new GaborFeature(gabor, new LuminanceFeature()));
    }
    PRINT_DEBUG("Dimensions = %d", features->getDimentions());
}

void readColorClassesFile(){
    cv::FileStorage file(colorClassesFilePath, cv::FileStorage::READ);
    file["colors"] >> classesColors;
    assert(classesColors.rows == numberOfClasses);
}
