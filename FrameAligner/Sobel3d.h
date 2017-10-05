#pragma once

#include <opencv2/opencv.hpp>

void sobel3d(cv::Mat m, cv::Mat& gx, cv::Mat& gy, cv::Mat& gz);