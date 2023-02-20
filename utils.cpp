#include "utils.h"

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
                strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}



cv::Mat preprocess_img(cv::Mat& img, int INPUT_H, int INPUT_W) {
    // std::cout<<"img2 0x0 B: "<<img.at<cv::Vec3f>(0,0)[0]<<std::endl;
    // std::cout<<"img2 0x0 G: "<<img.at<cv::Vec3f>(0,0)[1]<<std::endl;
    // std::cout<<"img2 0x0 R: "<<img.at<cv::Vec3f>(0,0)[2]<<std::endl;
    int w, h, x, y;
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


// cv::Mat curlImg(const char *img_url, int timeout=10)
// {
//     std::vector<uchar> stream;
//     auto curl_1 = std::chrono::steady_clock::now();
//     CURL *curl;
//     curl = curl_easy_init();
//     auto curl_2 = std::chrono::steady_clock::now();
//     std::cout<<"curl init time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_2-curl_1).count()<<"ms\n";
//     curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
    
//     curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    
//     curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
    
//     curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs, 
//     auto curl_6 = std::chrono::steady_clock::now();
//     CURLcode res = curl_easy_perform(curl); // start curl
//     auto curl_7 = std::chrono::steady_clock::now();
//     std::cout<<"curl perform time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_7-curl_6).count()<<"ms\n";
//     curl_easy_cleanup(curl); // cleanup
//     auto cur_8 = std::chrono::steady_clock::now();
//     std::cout<<"curlImage time is "<<std::chrono::duration_cast<std::chrono::milliseconds>(curl_8-curl_7).count()<<"ms\n";
//     return cv::imdecode(stream, -1); // 'keep-as-is'
// }
