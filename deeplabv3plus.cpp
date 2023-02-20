#include "deeplabv3plus.h"
#include "preprocess.h"
#include <string.h>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/matx.hpp>
// #include <opencv2/core/eigen.hpp>
#include <cmath>
#include <omp.h>

float e=0.00001;
const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

//curl writefunction to be passed as a parameter
// we can't ever expect to get the whole image in one piece,
// every router / hub is entitled to fragment it into parts
// (like 1-8k at a time),
// so insert the part at the end of our stream.
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    // auto write_1 = std::chrono::steady_clock::now();
    std::vector<uchar> *stream = (std::vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);

    // auto write_2 = std::chrono::steady_clock::now();
    // std::cout<<"write time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(write_2-write_1).count()<<"ms\n";
    return count;
}

void print(const std::vector<float> &v, Dims dimOut, std::string name)
{
    std::cout << name << ": (";
    for (int i = 0; i < dimOut.nbDims; ++i)
    {
        std::cout << dimOut.d[i] << ", ";
    }
    std::cout << "\b\b)" << std::endl;
    for (int b = 0; b < dimOut.d[0]; b++)
    {
        for (int h = 0; h < dimOut.d[1]; h++)
        {
            for (int w = 0; w < dimOut.d[2]; w++)
            {
                std::cout << std::fixed << std::setprecision(1) << std::setw(4) << v[(b * dimOut.d[0] + h) * dimOut.d[1] + w] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

//显示文件传输进度，dltotal代表文件大小，dlnow代表传输已经完成部分
//clientp是CURLOPT_PROGRESSDATA传入的值
int progress_callback(void *clientp, double dltotal, double dlnow, double ultotal, double ulnow){	
	if (dltotal != 0)
	{
		printf("%lf / %lf (%lf %%)\n", dlnow, dltotal, dlnow*100.0 / dltotal);
	}	
	return 0;
}

//load iamges from url
cv::Mat Inference::curlImg(const char *img_url, int timeout)
{
    auto curl_6 = std::chrono::steady_clock::now();
    std::vector<uchar> stream;
    // auto curl_1 = std::chrono::steady_clock::now();
    CURL *curl;
    curl_global_init(CURL_GLOBAL_ALL);
    curl = curl_easy_init();
    if(curl){
        curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
        /* Switch on full protocol/debug output while testing */
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
        // curl_easy_setopt(curl, CURLOPT_BUFFERSIZE,524288L);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 100000); // timeout if curl_easy hangs, 

    //     //实现下载进度
	// curl_easy_setopt(curl, CURLOPT_NOPROGRESS, false);
	// curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_callback);
	// curl_easy_setopt(curl, CURLOPT_PROGRESSDATA, nullptr);

       
        CURLcode res = curl_easy_perform(curl); // start curl
        while(res!=CURLE_OK){
            std::cout<<"failed to perform curl: "<<curl_easy_strerror(res)<<std::endl;
            res = curl_easy_perform(curl);
        }
        
        // std::cout<<"curl perform time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_7-curl_6).count()<<"ms\n";
        //check stream
        // std::cout<<"stream size"<<stream.size()<<std::endl;
        
        // auto curl_8 = std::chrono::steady_clock::now();
        // // std::cout<<"curl clean up time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_8-curl_7).count()<<"ms\n";
        // cv::Mat result = cv::imdecode(stream, -1); // 'keep-as-is'
        // auto curl_9 = std::chrono::steady_clock::now();
        // std::cout<<"curl stream to mat time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_9-curl_8).count()<<"ms\n";
    }else{
        std::cout<<"curl init failed\n";
    }
    auto curl_7 = std::chrono::steady_clock::now();
    std::cout<<"curl download time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_7-curl_6).count()<<"ms\n";
        
    curl_easy_cleanup(curl); // cleanup   
    curl_global_cleanup();
    cv::Mat imgGet = cv::imdecode(stream, -1); 
    auto curl_8 = std::chrono::steady_clock::now();
    std::cout<<"to Mat time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_8-curl_7).count()<<"ms\n";
       

    return imgGet;
}

//pading image to square
cv::Mat padImage(cv::Mat &img){
    int h = img.rows;
    int w = img.cols;
    int edge = h>w ? h:w;
    cv::Mat result(edge, edge, CV_8UC3, cv::Scalar(0, 0, 0));
    if(h>w){
        int x = (edge-w)/2;
        std::cout<<"start point: "<<x<<" "<< 0<<std::endl;
        img.copyTo(result(cv::Rect(x,0,w,h)));
    }
    else{
        int y = (edge-h)/2;
        std::cout<<"start point: "<<0<<" "<< y<<std::endl;
        img.copyTo(result(cv::Rect(0,y,w,h)));
    }
    return result;
}


bool check(cv::Mat &img, float* output){
   int i=0;
        for(int row=0; row<img.rows;++row)
        {
            float* uc_pixel = (float*)(img.data + row * img.step[0]);
            for(int col = 0; col < img.cols; ++col) 
            {
                // cv::Vec3f uc_pixel = img.at<cv::Vec3f>(row,col);
                if(fabs(output[i]-(float)(uc_pixel[0]))>e){
                    std::cout<<output[i]<< ' '<<(float)(uc_pixel[0]);
                    std::cout<<"wrong position: R "<<row<<' '<<col<<std::endl;
                    return false;
                }
                else if(fabs(output[i+1*img.rows*img.cols] != (float)(uc_pixel[1]))>e){
                     std::cout<<output[i+1*img.rows*img.cols]<< ' '<<(float)(uc_pixel[1]);
                    std::cout<<"wrong position: G "<<row<<' '<<col<<std::endl;
                    return false;
                }
                else if(fabs(output[i+2*img.rows*img.cols] != (float)(uc_pixel[2]))>e){
                     std::cout<<output[i+2*img.rows*img.cols]<< ' '<<(float)(uc_pixel[2]);
                    std::cout<<"wrong position: B "<<row<<' '<<col<<std::endl;
                   return false;
                }
                uc_pixel += 3;
                ++i;
            }
        }
    return true;;
}

cv::Mat test_print_output(int* output)
{ 
    int h= INPUT_H, w= INPUT_W;
    int x=0,  y=0;
    cv::Mat resA(h , w, CV_8UC1);
    // cv::Mat resB(h , w, CV_32FC1);
    // cv::Mat resC(h , w, CV_32FC1);
    std::cout<<h<<' '<<w<<std::endl;
    std::cout<<"copy output data..."<<std::endl;
    // #pragma omp parallel for
    for(int i = y; i<h+y; ++i)
    for(int j = x; j<w+x; ++j)
    {   
        resA.at<uchar>(i-y,j-x)=*(output+i*OUTPUT_W+j);
        // resB.at<float>(i-y,j-x)=*(output+OUTPUT_H*OUTPUT_W+i*OUTPUT_W+j);
        // resC.at<float>(i-y,j-x)=*(output+2*OUTPUT_H*OUTPUT_W+i*OUTPUT_W+j);
    }
    // std::vector<cv::Mat> channels;
    // channels.push_back(resC);
    // channels.push_back(resB);
    // channels.push_back(resA);
    // cv::Mat res;
    // cv::merge(channels, res);
    return resA;
}


// void saveCVS(cv::Mat &img, std::string fileName)
// {
    
//     std::ofstream file(fileName);
//     if(!file.is_open()){
//         std::cout<<"can't open the file\n";
//         return;
//     }
//     int row = img.rows;
//     int col = img.cols;
//     std::cout<<row<<' '<<col<<std::endl;
//     Eigen::MatrixXf m(row, col);
//     cv::cv2eigen(img,m);
//     std::cout<<m.rows()<<' '<<m.cols()<<std::endl;
//     // file<<m0.format(CSVFormat);
//     file<<m;
//     file.close();
// }

void saveArray(float* arr, int &w, int &h, int c, std::string filename)
{
    std::ofstream outFile; // 创建流对象
	outFile.open(filename, std::ios::out); // 打开文件
	int i=c*w*h;
    for(int k=0;k<h;++k){
        for(int j=0;j<w;++j)
		{
			outFile << arr[i+k*w+j]<<',' ;
		}
		outFile << std::endl;
	}
	outFile.close(); // 关闭文件
}

bool  checkOutput(float* output, std::vector<int> &mask_output)
{
    std::cout<<"Checking output ....\n";
    cv::Mat mask_mat(OUTPUT_H,OUTPUT_W, CV_8UC1);
    //parse float* to cv::mat
    for(int i=0;i<OUTPUT_H;++i)
    for(int j=0;j<OUTPUT_W;++j){
        float a = *(output+i*OUTPUT_W+j); 
        float b = *(output+OUTPUT_H*OUTPUT_W+i*OUTPUT_W+j);
        if(a>b)
            mask_mat.at<uchar>(i,j) = 0;
        else
            mask_mat.at<uchar>(i,j) = 1;
    }
    

    int r = mask_mat.rows;
    int c = mask_mat.cols;
    // std::cout<<"rows: "<<r<<" cols: "<<c<<std::endl;
    // std::cout<<"cuda output size: "<<mask_output.size()<<std::endl;
    // cv::Mat mat = cv::Mat(mask_output).clone();//将vector变成单列的mat
    // std::cout<<"-----------2-----------\n";
	// cv::Mat dest = mat.reshape(mask_mat.channels(), r);
    // std::cout<<dest.type()<<std::endl;
    cv::Mat dest(OUTPUT_H,OUTPUT_W, CV_8UC1);
    for(int i=0;i<OUTPUT_H;++i)
    for(int j=0;j<OUTPUT_W;++j){
        dest.at<uchar>(i,j) = mask_output[i*OUTPUT_W+j];
    }
    

    // cv::imwrite("/home/ylc/GMP/trt7/baresoil_v1/baresoil-v1/build/output/cudaOut.jpg", dest);
    // cv::imwrite("/home/ylc/GMP/trt7/baresoil_v1/baresoil-v1/build/output/cvOut.jpg", mask_mat);

    // std::cout<<"-----------3-----------\n";
    cv::Mat out;
    cv::bitwise_xor(mask_mat, dest, out);
    // std::cout<<"-----------4-----------\n";
    cv::Scalar ss = sum(out);
    std::cout<<ss[0]<<std::endl;
    if(ss[0]==0){
        std::cout<<"Congratulations! The output is exactly correct.\n";
        return true;
    }
    else
        std::cout<<"Sorry, the output is not right.\n";

    return false;

}

cv::Mat Inference::processCuMat(std::vector<int> &mask_output, int oh, int ow)
{
    cv::Mat dest(OUTPUT_H,OUTPUT_W, CV_8UC1);
    for(int i=0;i<OUTPUT_H;++i)
    for(int j=0;j<OUTPUT_W;++j){
        dest.at<uchar>(i,j) = mask_output[i*OUTPUT_W+j];
    }
    
    cv::Mat result(oh , ow, CV_8UC1);
    cv::resize(dest, result, cv::Size(ow,oh), 0, 0, cv::INTER_LINEAR);

    // cv::imwrite("/home/ylc/GMP/trt7/baresoil_v1/baresoil-v1/build/output/cumask.jpg", dest*255);
    return result;
}

cv::Mat Inference::preprocess(cv::Mat& img) {
    // int w, h, x, y;
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        w = INPUT_W;
        h = r_w * img.rows;
        x = 0;
        y = (INPUT_H - h) / 2;
    } else {
        w = r_h* img.cols;
        h = INPUT_H;
        x = (INPUT_W - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    // cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(img,re, re.size(), 0, 0, cv::INTER_LINEAR);

    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    // std::cout<<"img2 start point: "<<x<<' '<<y<<std::endl;
    // std::cout<<"B: "<<out.at<cv::Vec3u>(128,0)[0]<<std::endl;
    // std::cout<<"G: "<<out.at<cv::Vec3u>(128,0)[1]<<std::endl;
    // std::cout<<"R: "<<out.at<cv::Vec3u>(128,0)[2]<<std::endl;
    //减均值，除方差
    // auto data_start = std::chrono::system_clock::now();
    std::vector<float> mean{103.53, 116.28, 123.675};  //BGR
    std::vector<float> std{ 57.375, 57.12, 58.395};    //BGR
    cv::Mat norm_result;
    std::vector<cv::Mat> channels(3);
    std::vector<cv::Mat> channels_rgb(3);
    cv::split(out,channels);
    // std::cout<<"B: "<<channels[0].at<float>(128,0)<<std::endl;
    // std::cout<<"G: "<<channels[1].at<float>(128,0)<<std::endl;
    // std::cout<<"R: "<<channels[2].at<float>(128,0)<<std::endl;
    //convertTo(dst, type, scale, shift)
    for(auto i=0;i<channels.size();++i){
        // std::cout<<channels[i].at<uchar>(0,0)<<std::endl;
        channels[i].convertTo(channels_rgb[2-i], CV_32FC1, 1.0/std[i], (0.0-mean[i])/std[i]);
        
    }
    // std::cout<<"R: "<<channels_rgb[0].at<float>(128,0)<<std::endl;
    // std::cout<<"G: "<<channels_rgb[1].at<float>(128,0)<<std::endl;
    // std::cout<<"B: "<<channels_rgb[2].at<float>(128,0)<<std::endl;
    cv::merge(channels_rgb,norm_result);
    // auto data_end = std::chrono::system_clock::now();
    // std::cout<<"data opencv process cost "<<std::chrono::duration_cast<std::chrono::milliseconds>(data_end - data_start).count()<< "ms" <<std::endl;

    // float p1 = norm_result.at<cv::Vec3f>(128,0)[0];
    // float p2 = norm_result.at<cv::Vec3f>(128,0)[1];
    // float p3 = norm_result.at<cv::Vec3f>(128,0)[2];

    // std::cout<<"R: "<<p1<<std::endl;
    // std::cout<<"G: "<<p2<<std::endl;
    // std::cout<<"B: "<<p3<<std::endl;
    
    return norm_result;
}

std::vector<std::vector<cv::Point>> Inference::processCuOutput(std::vector<int> &mask_output, int oh, int ow)
{
    cv::Mat dest(OUTPUT_H,OUTPUT_W, CV_8UC1);
    for(int i=0;i<OUTPUT_H;++i)
    for(int j=0;j<OUTPUT_W;++j){
        dest.at<uchar>(i,j) = mask_output[i*OUTPUT_W+j];
    }
    
    cv::Mat img_resized = dest(cv::Rect(x,y,w,h));

    cv::Mat result(oh , ow, CV_8UC1);
    cv::resize(img_resized, result, cv::Size(ow,oh), 0, 0, cv::INTER_LINEAR);

    // cv::imwrite("/home/a119/Documents/GMP/baresoil/baresoil-v4/build/output/mask.jpg", dest*255);

    double minArea = 0.001*oh*ow;
    std::vector<std::vector<cv::Point>> contours_reserve;
    if(cv::countNonZero(result))
    {
        std::vector<std::vector<cv::Point>> contours;
        
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(result,contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::vector<std::vector<cv::Point>> approx_contours(contours.size());
        // std::cout<<contours.size()<<std::endl;
	    for(int i=0;i<contours.size();++i){
		    double area = cv::contourArea(contours[i]);  //获得轮廓面积
            // std::cout<<i<<' '<<area<<std::endl;
		    if (area<minArea)    
		        continue;  //过滤较小面积的轮廓 
            else{
                cv::approxPolyDP(cv::Mat(contours[i]), approx_contours[i], 30, true);
                contours_reserve.emplace_back(approx_contours[i]);
            }
	    }
    }

    return contours_reserve;
}

/**
std::vector<std::vector<cv::Point>> processCuOutput(std::vector<float> &mask_output, int oh, int ow)
{
    cv::Mat dest(OUTPUT_H,OUTPUT_W, CV_8UC1);
    for(int i=0;i<OUTPUT_H;++i)
    for(int j=0;j<OUTPUT_W;++j){
        dest.at<uchar>(i,j) = static_cast<int>(mask_output[i*OUTPUT_W+j]);
    }
    
    cv::Mat result(oh , ow, CV_8UC1);
    cv::resize(dest, result, cv::Size(ow,oh), 0, 0, cv::INTER_LINEAR);
    cv::imwrite("/workspace/projects/baresoil/baresoil-v3/output/finalOut.jpg", result*255);

    double minArea = 0.001*oh*ow;
    std::vector<std::vector<cv::Point>> contours_reserve;
    if(cv::countNonZero(result))
    {
        std::vector<std::vector<cv::Point>> contours;
        
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(result,contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::vector<std::vector<cv::Point>> approx_contours(contours.size());
        // std::cout<<contours.size()<<std::endl;
	    for(int i=0;i<contours.size();++i){
		    double area = cv::contourArea(contours[i]);  //获得轮廓面积
            // std::cout<<i<<' '<<area<<std::endl;
		    if (area<minArea)    
		        continue;  //过滤较小面积的轮廓 
            else{
                cv::approxPolyDP(cv::Mat(contours[i]), approx_contours[i], 30, true);
                contours_reserve.emplace_back(approx_contours[i]);
            }
	    }
    }

    return contours_reserve;

    // return resA_re;
}
*/


//单张图像数据处理
std::vector<std::vector<cv::Point>>  Inference::inferMat(std::string image_file_name){
        std::cout<<image_file_name<<std::endl;
       
        // cv::Mat orign_img = cv::imread(image_file_name);
        const char* imgdata = image_file_name.c_str();
        // auto curl_start = std::chrono::steady_clock::now();
        cv::Mat orign_img = curlImg(imgdata);
        std::vector<float> data(BATCH_SIZE * 3 * INPUT_H * INPUT_W);
        // float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        // float prob[BATCH_SIZE * OUTPUT_SIZE];

        // auto curl_end = std::chrono::steady_clock::now();
        // std::cout<<"curltime is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_end-curl_start).count()<<"ms\n";
        // cv::imwrite("/workspace/projects/TensorRT_EVN/projects/git/baresoil-v4/build/output/"+image_file_name.substr(9,13)+".jpg",orign_img);
        /**
         * 预处理过程一，CPU
        */
        IExecutionContext* context = engine->createExecutionContext();
        // auto pre_start = std::chrono::steady_clock::now();
        // cv::Mat resize_img= scaleImage(orign_img);
        cv::Mat img = preprocess(orign_img);

        // auto pre_end = std::chrono::steady_clock::now();
        // std::cout<<"preprocess time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(pre_end-pre_start).count()<<"ms\n";

        //将Mat数据按通道赋值给数组， 数组大小为W*H*3 
        auto copy_start = std::chrono::steady_clock::now();
        int b=0, i=0;
        for(;b<BATCH_SIZE;++b)
        {
        for (int row = 0; row < INPUT_H; ++row) {
            float* uc_pixel = (float*)(img.data + row * img.step[0]);
            for (int col = 0; col < INPUT_W; ++col) {
                data[b * 3 * INPUT_H * INPUT_W + i] = (float)(uc_pixel[0]);
                data[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)(uc_pixel[1]);
                data[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)(uc_pixel[2]);
                uc_pixel += 3;
                ++i;
            }
        }
        }
        // auto copy_end= std::chrono::steady_clock::now();
        // std::cout<<"preprocess time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(pre_start-copy_start).count()<<"ms\n";

        int inputSize =  3 * INPUT_H * INPUT_W ;
        // auto big_start = std::chrono::steady_clock::now();
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        void* buffers[2];
        CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&buffers[outputIndex],  BATCH_SIZE * OUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(float)));
                
        CUDA_CHECK(cudaMemcpyAsync(buffers[0], data.data(), BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        

        /**
         * kernel version
        */
       /**
        auto pre_start = std::chrono::steady_clock::now();
        uint8_t* img_host = nullptr;
        uint8_t* img_device = nullptr;
        int img_size = orign_img.cols*orign_img.rows;
        // prepare input data cache in pinned memory 
        CUDA_CHECK(cudaMallocHost((void**)&img_host, img_size));
        // prepare input data cache in device memory
        CUDA_CHECK(cudaMalloc((void**)&img_device, img_size));
    
        //copy data to pinned memory
        memcpy(img_host, orign_img.data, img_size);
        CUDA_CHECK(cudaMemcpyAsync(img_device, img_host, img_size, cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device,orign_img.cols,orign_img.rows,static_cast<float*>(buffers[inputIndex]),INPUT_W,INPUT_H,stream);
        auto pre_end = std::chrono::steady_clock::now();
        std::cout<<"preprocess time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(pre_end-pre_start).count()<<"ms\n";
        
        auto big_start = std::chrono::steady_clock::now();
        */
        context->enqueue(BATCH_SIZE, (void**)buffers, stream, nullptr);

        //argmax cuda process
        int* cu_output=NULL;
        int mask_size = 1*INPUT_H*INPUT_W;
        std::vector<int> mask_output(mask_size,0.0f);
        CUDA_CHECK(cudaMalloc((void**)&cu_output, sizeof(int)*mask_size));
        argmax(static_cast<float*>(buffers[1]), cu_output, OUTPUT_H, OUTPUT_W, CLASS_NUM);
        CUDA_CHECK(cudaMemcpy(mask_output.data(),cu_output, sizeof(int)*mask_size,cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree(cu_output));
        CUDA_CHECK(cudaFree(buffers[inputIndex]));
        CUDA_CHECK(cudaFree(buffers[outputIndex]));
        // CUDA_CHECK(cudaMemcpyAsync(prob, buffers[1], BATCH_SIZE * OUTPUT_SIZE* sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        // auto big_end = std::chrono::steady_clock::now();
        // std::cout<<"pure inference time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(big_end-big_start).count()<<"ms\n";
        
        // auto post_start = std::chrono::steady_clock::now();
        // auto mask = processCuMat(mask_output,orign_img.rows, orign_img.cols);
        auto mask = processCuOutput(mask_output, orign_img.rows, orign_img.cols);
        // auto post_end = std::chrono::steady_clock::now();
        // std::cout<<"pure postprocess time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(post_end-post_start).count()<<"ms\n";
        context->destroy();
        cudaStreamDestroy(stream);
        return mask; 
}


Inference::Inference(std::string engine_file_name){
        std::ifstream file(engine_file_name, std::ios::binary);
        // curl_global_init(CURL_GLOBAL_ALL);
        // this->curl = curl_easy_init();
        if (!file.good()) {
            std::cerr << "read " << engine_file_name << " error!" << std::endl;
        }
        else{
            try{
                std::cout<<"------1-1--------\n";
                // char *trtModelStream = nullptr;
                size_t size = 0;
                file.seekg(0, file.end);
                size = file.tellg();
                file.seekg(0, file.beg);
                // trtModelStream = new char[size];
                // assert(trtModelStream);
                std::vector<char> trtModelStream(size);
                file.read(trtModelStream.data(), size);
                file.close();
                if(trtModelStream.size() == 0)
                {
                    std::cout << "Failed getting serialized engine!" << std::endl;
                    return;
                }
                file.close();
                std::cout << "Succeeded getting serialized engine!" << std::endl;

                runtime = createInferRuntime(gLogger);
                assert(runtime != nullptr);
                std::cout<<"------1-3--------\n";
                this->engine = runtime->deserializeCudaEngine(trtModelStream.data(), size);
                 if (engine == nullptr)
                {
                    std::cout << "Failed building engine!" << std::endl;
                    return;
                }
                std::cout << "Succeeded building engine!" << std::endl;
                runtime->destroy();

                assert(engine != nullptr); 
                std::cout<<"------1-4--------\n";
            
                // this->context = this->engine->createExecutionContext();
                // assert(this->context != nullptr);
                assert(engine->getNbBindings() == 2);
                // std::cout<<"------1-5--------\n";
                // // delete[] trtModelStream;
               

                inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
                outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
                // CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
                // CUDA_CHECK(cudaMalloc(&buffers[outputIndex],  BATCH_SIZE * OUT_CHANNEL * OUTPUT_H * OUTPUT_W * sizeof(float)));
                // // Create stream
                // cudaStream_t stream;
                // CUDA_CHECK(cudaStreamCreate(&stream));
                // std::cout<<"------1-5--------\n";

            }
            catch(...){
                    std::cerr<<"Initialization failed.\n";
            }
        }
        
    }

Inference::~Inference(){
    std::cout<<"Exit Inference\n";
    // context->destroy();
    engine->destroy();
    // runtime->destroy();
    // curl_easy_cleanup(curl); // cleanup   
    // curl_global_cleanup();
    
}


cv::Mat Inference::scaleImage(cv::Mat& img){
    float r_w = INPUT_W / (img.cols*1.0);
    float r_h = INPUT_H / (img.rows*1.0);
    if (r_h > r_w) {
        this->w = INPUT_W;
        this->h = r_w * img.rows;
        this->x = 0;
        this->y = (INPUT_H - this->h) / 2;
    } else {
        this->w = r_h* img.cols;
        this->h = INPUT_H;
        this->x = (INPUT_W - this->w) / 2;
        this->y = 0;
    }
    cv::Mat re(this->h, this->w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_CUBIC);
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(0, 0, 0));
    re.copyTo(out(cv::Rect(this->x, this->y, re.cols, re.rows)));

    return out;
}

//postprocess for segmentations
std::vector<std::vector<cv::Point>> Inference::process_output(float* output, int& oh, int& ow)
{
    auto mat_start= std::chrono::steady_clock::now();
    cv::Mat result = process_Mat(output, oh, ow);
    std::cout<<"saving mat...\n";
    if(!cv::imwrite("/workspace/projects/TensorRT_EVN/projects/git/baresoil-v4/build/output/maskMat.jpg", result))
       std::cout<<"failed to save img\n";
    auto mat_end= std::chrono::steady_clock::now();
    std::cout<<"process mat time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(mat_end-mat_start).count()<<"ms\n";

    double minArea = 0.001*oh*ow;
    
    /* get contours */
    /* result format{"result":{"bare soil",.segmentation points},"shape":"poloygon"}  */
    std::vector<std::vector<cv::Point>> contours_reserve;
    if(cv::countNonZero(result))
    {
        std::vector<std::vector<cv::Point>> contours;
        
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(result,contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        std::vector<std::vector<cv::Point>> approx_contours(contours.size());
        // std::cout<<contours.size()<<std::endl;
	    for(int i=0;i<contours.size();++i){
		    double area = cv::contourArea(contours[i]);  //获得轮廓面积
            // std::cout<<i<<' '<<area<<std::endl;
		    if (area<minArea)    
		        continue;  //过滤较小面积的轮廓 
            else{
                cv::approxPolyDP(cv::Mat(contours[i]), approx_contours[i], 30, true);
                contours_reserve.emplace_back(approx_contours[i]);
            }
	    }
    }

    return contours_reserve;
}

cv::Mat Inference::process_Mat(float* output, int& oh, int& ow)
{

    float r_w = INPUT_W / (ow*1.0);
    float r_h = INPUT_H / (oh*1.0);
    if (r_h > r_w) {
        this->w = INPUT_W;
        this->h = r_w * oh;
        this->x = 0;
        this->y = (INPUT_H - this->h) / 2;
    } else {
        this->w = r_h* ow;
        this->h = INPUT_H;
        this->x = (INPUT_W - this->w) / 2;
        this->y = 0;
    }

    // // auto start0 = std::chrono::steady_clock::now();
    // cv::Mat resA(h , w, CV_8UC1);
    cv::Mat resA(h , w, CV_32FC1);
    cv::Mat resB(h , w, CV_32FC1);
    std::cout<<h<<' '<<w<<std::endl;
    std::cout<<"copy output data..."<<std::endl;
    // // #pragma omp parallel for
    for(int i = y; i<h+y; ++i)
    for(int j = x; j<w+x; ++j)
    {   
        resA.at<float>(i-y,j-x)=*(output+i*OUTPUT_W+j);
        resB.at<float>(i-y,j-x)=*(output+OUTPUT_H*OUTPUT_W+i*OUTPUT_W+j);
    }
    std::cout<<"copy output data finished."<<std::endl;
    // auto end0 = std::chrono::steady_clock::now();
    // std::cout<<"float to mat "<<std::chrono::duration_cast<std::chrono::milliseconds>(end0-start0).count()<<"ms\n";
    
    auto start2 = std::chrono::steady_clock::now();
    cv::Mat resA_re(oh , ow, CV_32FC1);
    cv::Mat resB_re(oh , ow, CV_32FC1);
    std::cout<<oh<<' '<<ow<<std::endl;
    // cv::Mat resA_re(oh , ow, CV_8UC1);
    cv::resize(resA, resA_re, cv::Size(ow,oh), 0, 0, cv::INTER_LINEAR);

    cv::resize(resB, resB_re, cv::Size(ow,oh), 0, 0, cv::INTER_LINEAR);
    auto end2 = std::chrono::steady_clock::now();
    std::cout<<"resize mat "<<std::chrono::duration_cast<std::chrono::milliseconds>(end2-start2).count()<<"ms\n";
    
    cv::Mat result(oh, ow, CV_8UC1, cv::Scalar(0));
    // result = resA_re*255;
    
    auto start = std::chrono::steady_clock::now();
    float* p = nullptr;
    float* q = nullptr;
    // #pragma omp parallel for
    for(int i=0;i<oh;++i)
    {
        p = resA_re.ptr<float>(i);
        q = resB_re.ptr<float>(i);
        for(int j =0;j<ow;++j)
        {
            if(p[j]>=q[j]){
                result.at<uchar>(i,j)=0;
            }
            else{
                result.at<uchar>(i,j)=255;
            }
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::cout<<"01 mat "<<std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count()<<"ms\n";

    if(!cv::imwrite("/workspace/projects/TensorRT_EVN/projects/git/baresoil-v4/build/output/maskMat.jpg", result))
       std::cout<<"failed to save img\n";
    return result;
}

std::map<std::string, Weights> loadWeights(const std::string file){
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        // float* val = reinterpret_cast<float*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;

}


ILayer* decodeHeadLayer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, ITensor &input0, int outch, int *dilation, std::string lname, int dSize){
    //decode_head = image_pool+aspp_modules+bottleneck+c1_bottleneck+sep_bottleneck+conv_seg
   
    auto imagePoolLayer = imagePool(network, weightMap, input, 512, lname+".image_pool");
    // auto tmp = imagePoolLayer->getOutput(0);
    auto resize1 = network->addResize(*imagePoolLayer->getOutput(0));
    assert(resize1);
    resize1->setResizeMode(ResizeMode::kLINEAR);//bilinear 2D
    resize1->setAlignCorners(false);
    // constexpr char* value[] = { "aaa", "bbb", "ccc" };
    // const float *outdims[4];
    constexpr float outdims[] = {1, 128, 128 };
    resize1->setScales(outdims,3);  //退化为指针

    // resize1->setOutputDimensions(input.getDimensions());
    //aspp_modules
    // auto conv1 = convBlock(network, weightMap, input, outch, 1, lname+".0.conv.weight", lname+"");
    IActivationLayer* depthwiseConv[4]={nullptr};
    IActivationLayer* pointwiseConv[4]={nullptr};
    std::string aspp_name = lname + ".aspp_modules.";
    for( int i=0;i<dSize; ++i)
    {
        if(dilation[i]>1){
            depthwiseConv[i] = convBlock(network, weightMap, input, 2048, 3, aspp_name+std::to_string(i)+".depthwise_conv.conv.weight", aspp_name+std::to_string(i)+".depthwise_conv.bn", 1,  dilation[i],  dilation[i], 2048);
            pointwiseConv[i] = convBlock(network, weightMap, *depthwiseConv[i]->getOutput(0), 512, 1, aspp_name+std::to_string(i)+".pointwise_conv.conv.weight", aspp_name+std::to_string(i)+".pointwise_conv.bn");
        }
        else{
            pointwiseConv[i] = convBlock(network, weightMap, input, 512, 1, aspp_name+std::to_string(i)+".conv.weight", aspp_name+std::to_string(i)+".bn");
        }
    }
    ITensor* inputTensors1[] = {resize1->getOutput(0), pointwiseConv[0]->getOutput(0), pointwiseConv[1]->getOutput(0), pointwiseConv[2]->getOutput(0),
                               pointwiseConv[3]->getOutput(0)};

    auto cat1 = network->addConcatenation(inputTensors1, 5);

    auto bottleneck1 = convBlock(network, weightMap, *cat1->getOutput(0), 512, 3, lname+".bottleneck.conv.weight", lname+".bottleneck.bn", 1, 1) ;

    auto c1_bottleneck = convBlock(network,weightMap, input0, 48, 1, lname+".c1_bottleneck.conv.weight", lname+".c1_bottleneck.bn");
    
    auto resize2 = network->addResize(*bottleneck1->getOutput(0));
    resize2->setResizeMode(ResizeMode::kLINEAR);//bilinear 2D
    resize2->setAlignCorners(false);
    constexpr float value[] = {1, 2, 2 };
    resize2->setScales(value,3);

    ITensor* inputTensors2[] = {resize2->getOutput(0), c1_bottleneck->getOutput(0)};
    auto cat2 = network->addConcatenation(inputTensors2, 2);

    auto sep_bottleneck = sepBottleNeck(network, weightMap, *cat2->getOutput(0), 512, lname+".sep_bottleneck");

    auto cov_seg = network->addConvolutionNd(*sep_bottleneck->getOutput(0), CLASS_NUM, DimsHW{1,1}, weightMap[lname+".conv_seg.weight"], weightMap[lname+".conv_seg.bias"]);

    return cov_seg;
}

ICudaEngine* createEngine_d(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, const std::string weightFile) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    
    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3,INPUT_H,INPUT_W});
    assert(data);
    //Load model weights from weights file
    std::map<std::string, Weights> weightMap = loadWeights(weightFile);
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    /*----------deeplabv3+ backbone----------*/
    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 32, DimsHW{3, 3}, weightMap["backbone.stem.0.weight"], emptywts);
    // IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{7, 7}, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{2, 2});
    conv1->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.stem.1", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), 32, DimsHW{3, 3}, weightMap["backbone.stem.3.weight"], emptywts);
    assert(conv2);
    conv2->setStrideNd(DimsHW{1, 1});
    conv2->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), "backbone.stem.4", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), 64, DimsHW{3, 3}, weightMap["backbone.stem.6.weight"], emptywts);
    assert(conv3);
    conv3->setStrideNd(DimsHW{1, 1});
    conv3->setPaddingNd(DimsHW{1, 1});

    IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), "backbone.stem.7", 1e-5);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer* relu3 = network->addActivation(*bn3->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    auto maxPool1 = network->addPoolingNd(*relu3->getOutput(0), PoolingType::kMAX, DimsHW{3,3});
    maxPool1->setPaddingNd(DimsHW{1,1});  
    maxPool1->setStrideNd(DimsHW{2,2});

    IActivationLayer* x = bottleneck(network, weightMap, *maxPool1->getOutput(0), 64, 64, 1, 1, "backbone.layer1.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, 1, "backbone.layer1.1.");
    IActivationLayer* backBoneStage1 = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1, 1, "backbone.layer1.2.");

    x = bottleneck(network, weightMap, *backBoneStage1 ->getOutput(0), 256, 128, 2, 1, "backbone.layer2.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1, 1, "backbone.layer2.3.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 1, 1, "backbone.layer3.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 2, "backbone.layer3.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 2, "backbone.layer3.2.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 2, "backbone.layer3.3.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 2, "backbone.layer3.4.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1, 2, "backbone.layer3.5.");

    x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 1, 2, "backbone.layer4.0.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, 4, "backbone.layer4.1.");
    x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1, 4, "backbone.layer4.2.");

    //build decode head
    //the outpts of backbone is {backBoneStage1,backBoneStage2,backBoneStage3,backBoneStage4}
    //in_index=3, so the input to decode head is the ouput of backBoneStage4
    //set dilation
    int dilation[4]={1,12,24,36};
    auto decoderHead = decodeHeadLayer(network, weightMap, *x->getOutput(0), *backBoneStage1->getOutput(0), 3, dilation, "decode_head", 4);
    //auxiliary head didn't participate in inference forward
    //resize output to image input scale
    auto resize = network->addResize(*decoderHead->getOutput(0));
    resize->setResizeMode(ResizeMode::kLINEAR);//bilinear 2D
    resize->setAlignCorners(false);
    // Dims in = decoderHead->getOutput(0)->getDimensions();
    // Dims32 outdims[] = {CLASS_NUM, 1024, 1024 };
    resize->setOutputDimensions(Dims3{CLASS_NUM,INPUT_H,INPUT_W});


    resize->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*resize->getOutput(0));

    //build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(64*(1<<20)); //16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "/data/wsl/VOC/JPEGImages/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    network->destroy();

    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, std::string weightFile) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = createEngine_d(maxBatchSize, builder, config, DataType::kFLOAT, weightFile);
    assert(engine != nullptr);
    
    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
    config->destroy();
}






