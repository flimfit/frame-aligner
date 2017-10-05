#pragma once
#include <opencv2/opencv.hpp>

#define X 2
#define Y 1
#define Z 0


cv::Mat extractSlice(const cv::Mat& m, int slice);
int area(const cv::Mat& m);

void writeScaledImage(const std::string&, const cv::Mat& m);

double correlation(cv::Mat &image_1, cv::Mat &image_2, cv::Mat &mask);
void interpolatePoint3d(const std::vector<cv::Point3d>& Ds, std::vector<cv::Point3d>& D);