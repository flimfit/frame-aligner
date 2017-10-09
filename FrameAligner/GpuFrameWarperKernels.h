
#pragma once

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class GpuFrame
{
public:
   GpuFrame(cv::Mat frame);
   ~GpuFrame();

   bool isSame(const cv::Mat& frame_) const { return (frame.data == frame_.data); }

protected:
   cv::Mat frame;
   cudaArray *cu_array;
   int texture;
};
