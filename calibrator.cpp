#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <opencv2/dnn/dnn.hpp>
#include "calibrator.h"
#include "utils.h"
#include "cuda_utils.h"
#include <iterator>

Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batchsize, int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    input_count_ = 3 * input_w * input_h * batchsize;
    CUDA_CHECK(cudaMalloc(&device_input_, input_count_ * sizeof(float)));
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator2::getBatchSize() const TRT_NOEXCEPT
{
    return batchsize_;
}


bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nbBindings) TRT_NOEXCEPT
{
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
        input_imgs_.push_back(pr_img);
    }
    img_idx_ += batchsize_;

    // auto start = std::chrono::system_clock::now();
    // float data[batchsize_*3*input_w_*input_h_];
 
    // std::cout<<"vector size: "<<input_imgs_.size()<<std::endl;
    // for(int b=0; b<input_imgs_.size(); ++b){
    //     cv::Mat img = input_imgs_[b];
    //     int i=0;
    //     for (int row = 0; row < input_h_; ++row) {
    //             float* uc_pixel = (float*)(img.data + row * img.step[0]);
    //             for (int col = 0; col < input_w_; ++col) {
    //                 data[b * 3 * input_w_*input_h_ + i] = (float)(uc_pixel[0]);
    //                 data[b * 3 * input_w_*input_h_ + i + input_w_*input_h_] = (float)(uc_pixel[1]);
    //                 data[b * 3 * input_w_*input_h_ + i + 2 * input_w_*input_h_] = (float)(uc_pixel[2]);
    //                 uc_pixel += 3;
    //                 ++i;
    //             }
    //         }

    // }
    // auto end = std::chrono::system_clock::now();
    // std::cout<<"batch data copy: "<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    //blobFromImages: 可以实现减均值，交换通道，缩放，裁剪等
    cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0), false, false, CV_32F);
    // std::cout<<sizeof(input_imgs_)<<std::endl;
    // std::cout<<input_count_ * sizeof(float)<<std::endl;
    CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMemcpy(device_input_, &data, input_count_ * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    
    return true;
}


const void* Int8EntropyCalibrator2::readCalibrationCache(size_t& length) TRT_NOEXCEPT
{
    std::cout << "reading calib cache: " << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();
    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, size_t length) TRT_NOEXCEPT
{
    std::cout << "writing calib cache: " << calib_table_name_ << " size: " << length << std::endl;
    std::ofstream output(calib_table_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}