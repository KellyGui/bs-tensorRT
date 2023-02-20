#include "common.h"

using namespace nvinfer1;

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* bottleneck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, int dilation, std::string lname) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{stride, stride});
    conv2->setPaddingNd(DimsHW{dilation, dilation});
    conv2->setDilation(DimsHW{dilation, dilation});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
    assert(conv3);

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

    IElementWiseLayer* ew1;
    if (stride != 1 || inch != outch * 4) {
        IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv4);
        conv4->setStrideNd(DimsHW{stride, stride});

        IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
    } else {
        ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu3 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu3);
    return relu3;
}


IActivationLayer*  steamBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, int s, std::string lnameCon, std::string lnameBn){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };
    int p = ksize / 3;
    IConvolutionLayer* conv1 = network->addConvolutionNd(input,outch,DimsHW{ksize,ksize}, weightMap[lnameCon], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW({s,s}));
    conv1->setPaddingNd(DimsHW({p,p}));
      
    IScaleLayer* bn1 = addBatchNorm2d(network,weightMap, *conv1->getOutput(0),lnameBn, 1e-5);
    assert(bn1);
    
    //ReLU
    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
    
}


IActivationLayer*  convBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, int ksize, std::string lnameCon, std::string lnameBn, int s, int p, int d, int groups){
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input,outch,DimsHW{ksize,ksize}, weightMap[lnameCon], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW({s,s}));
    conv1->setPaddingNd(DimsHW{p,p});
    conv1->setDilation(DimsHW{d,d});
    conv1->setNbGroups(groups);
      
    IScaleLayer* bn1 = addBatchNorm2d(network,weightMap, *conv1->getOutput(0),lnameBn, 1e-5);
    assert(bn1);

    //ReLU
    auto relu = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu);

    return relu;
    
}

ILayer* resNetBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, std::string lname, int expansion, ILayer *downSample){
    auto cv1 = convBlock(network, weightMap, input, outch, 1, lname+".conv1.weight", lname+".bn1"); //stride=1, padding=0
    auto cv2 = convBlock(network, weightMap, *cv1->getOutput(0), outch, 3, lname+".conv2.weight", lname+".bn2", s, 1);
    auto cv3 = convBlock(network, weightMap, *cv2->getOutput(0), outch*expansion, 1, lname+".conv3.weight", lname+".bn3");//stride =1, padding =0
    if(downSample!=nullptr){
        auto add = network->addElementWise(*cv3->getOutput(0),*downSample->getOutput(0), ElementWiseOperation::kSUM);
        auto relu = network->addActivation(*add->getOutput(0), ActivationType::kRELU);
        return relu;
    }
    auto add2 = network->addElementWise(*cv3->getOutput(0),input,ElementWiseOperation::kSUM);
    auto relu = network->addActivation(*add2->getOutput(0), ActivationType::kRELU);
    return relu;
}

ILayer* imagePool(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname){
    auto avgPool = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{128,128});
    assert(avgPool);
    auto conv1 = convBlock(network, weightMap, *avgPool->getOutput(0), outch, 1, lname+".1.conv.weight", lname+".1.bn");
    assert(conv1);
    return conv1;
}

ILayer* sepBottleNeck(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int outch, std::string lname){
    int inch = input.getDimensions().d[0];
    auto depthwiseConv1 = convBlock(network, weightMap, input, inch, 3, lname+".0.depthwise_conv.conv.weight", lname+".0.depthwise_conv.bn", 1, 1, 1, inch);
    auto pointwiseConv1 = convBlock(network, weightMap, *depthwiseConv1->getOutput(0), outch, 1, lname+".0.pointwise_conv.conv.weight", lname+".0.pointwise_conv.bn");
    inch = pointwiseConv1->getOutput(0)->getDimensions().d[0];
    auto depthwiseConv2 = convBlock(network, weightMap, *pointwiseConv1->getOutput(0), inch, 3, lname+".1.depthwise_conv.conv.weight", lname+".1.depthwise_conv.bn", 1, 1, 1, inch);
    auto pointwiseConv2 = convBlock(network, weightMap, *depthwiseConv2->getOutput(0), outch, 1, lname+".1.pointwise_conv.conv.weight", lname+".1.pointwise_conv.bn");
    return pointwiseConv2;
}
