#pragma once
#include <opencv2/opencv.hpp>

cv::Mat extractSlice(const cv::Mat& m, int slice);
int area(const cv::Mat& m);