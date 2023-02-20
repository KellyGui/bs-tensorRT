#ifndef DEEPLABV3PLUS_COMMON_H_
#define DEEPLABV3PLUS_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"

using namespace nvinfer1;

// inference on one GPU, the SyncBN==BN
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps);

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, int dilation, std::string lname);

IActivationLayer*  steamBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, std::string lnameCon, std::string lnameBn);


IActivationLayer*  convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lnameCon, std::string lnameBn, int s=1, int p=0, int d=1, int groups=1);

ILayer* resNetBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, std::string lname, int expansion, ILayer *downSample=nullptr);

ILayer* imagePool(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname);

ILayer* sepBottleNeck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname);

#endif