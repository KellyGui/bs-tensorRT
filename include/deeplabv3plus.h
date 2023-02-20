#include <iostream>
#include <chrono>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include "cuda_utils.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include "logging.h"
#include <cmath>
#include <curl/curl.h>
#include <curl/easy.h>
#include <opencv2/imgproc/types_c.h>
#include "calibrator.h"
#include "utils.h"
#include "argmax.h"


static Logger gLogger;
#define USE_FP16
// #define USE_INT8
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1


// For image resize, keep ratio = True 
//so that input size is not always be a square.
static const int INPUT_H = 1024;
static const int INPUT_W = 1024;
static const int OUTPUT_H = 1024;
static const int OUTPUT_W = 1024;
static const int OUT_CHANNEL = 2;
static const int CLASS_NUM = 2;
static const int OUTPUT_SIZE = OUT_CHANNEL*OUTPUT_H*OUTPUT_W;
// static const double minArea = 10000;

using namespace nvinfer1;

class Inference{
public:
    std::string engine_file_name;
    ICudaEngine* engine;
    // IExecutionContext* context;
    IRuntime* runtime;
    // cudaStream_t stream;
    // float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    // float prob[BATCH_SIZE * OUTPUT_SIZE];
    // void* buffers[2];
    int inputIndex;
    int outputIndex;
    int w, h, x, y;
    // CURL *curl;
    
    
    Inference()=delete;
    Inference(std::string engine_file_name);

    Inference(const Inference &other)=delete;//复制构造函数
    Inference& operator=(const Inference& other)=delete; // 拷贝赋值
    

    //单张图像数据处理
    std::vector<std::vector<cv::Point>>  inferSingle(std::string image_file_name);
    
    cv::Mat scaleImage(cv::Mat& img);

    std::vector<std::vector<cv::Point>>  inferMat(std::string image_file_name);

    cv::Mat process_Mat(float* output, int& oh, int& ow);
    cv::Mat processCuMat(std::vector<int> &mask_output, int oh, int ow);

    std::vector<std::vector<cv::Point>> process_output(float* output, int& oh, int& ow);

    std::vector<std::vector<cv::Point>> processCuOutput(std::vector<int> &mask_output, int oh, int ow);
    
    cv::Mat curlImg(const char *img_url, int timeout=10); //fetch image from url

    void outputTest(float* out);

    cv::Mat preprocess(cv::Mat& img);

    ~Inference();
    
};

std::map<std::string, Weights> loadWeights(const std::string file);
ILayer* steamLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, std::string lname, int ksize);
ILayer* decodeHeadLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, ITensor &input0, int outch, int *dilation, std::string lname, int dSize);

ICudaEngine* createEngine_d(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string weightFile);
ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string weightFile);
ICudaEngine* createEngine_test(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string weightFile);

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string weightFile);
void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize,  int inputSize, int outputSize);
// void saveArray(vector<float> arr, int &w, int &h, int c, std::string filename);
bool  checkOutput(float* mask_mat, std::vector<int> &mask_output);