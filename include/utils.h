#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include <vector>

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);
cv::Mat preprocess_img(cv::Mat& img, int INPUT_H, int INPUT_W);

#endif