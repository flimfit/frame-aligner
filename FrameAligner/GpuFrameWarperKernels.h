
#pragma once

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

class GpuFrame
{
public:
   GpuFrame(cv::Mat frame, int nD);
   ~GpuFrame();

   bool isSame(const cv::Mat& frame_) const { return (frame.data == frame_.data); }

   int getTextureId() { return texture; };

   float* error_image;
   float* error_sum;
   float3* D;
   int3 size;

protected:
   cv::Mat frame;
   cudaArray *cu_array;
   int texture;
};

struct GpuRange
{
   int begin;
   int end;
};

class GpuReferenceInformation
{
public:

   GpuReferenceInformation(const cv::Mat& reference, float3 offset, int nD, int range_max);
   ~GpuReferenceInformation();

   float* reference = nullptr;
   float3 *VI_dW_dp = nullptr;
   GpuRange *range = nullptr;
   float3 offset;
   int nD = 0;
   int range_max = 0;
};

double computeError(GpuFrame* frame, GpuReferenceInformation* gpu_ref);
