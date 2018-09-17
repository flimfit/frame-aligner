#pragma once

#include <opencv2/opencv.hpp>
#include "Cache.h"
//#include "Cache_impl.h"

template<>
Cache<cv::Mat>* Cache<cv::Mat>::instance;

template<>
size_t Cache<cv::Mat>::getSize(const cv::Mat& obj);
