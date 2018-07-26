#pragma once
#include <opencv2/opencv.hpp>

extern const int X;
extern const int Y;
extern const int Z;


cv::Mat extractSlice(const cv::Mat& m, int slice);
int area(const cv::Mat& m);

void writeScaledImage(const std::string&, const cv::Mat& m);

double correlation(cv::Mat &image_1, cv::Mat &image_2, cv::Mat &mask);
void interpolatePoint3d(const std::vector<cv::Point3d>& Ds, std::vector<cv::Point3d>& D);

void inpaint3d(const cv::Mat& input, const cv::Mat& mask, cv::Mat& output);